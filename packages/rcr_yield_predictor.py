import os
from typing import List
from aizynthfinder.chem import TemplatedRetroReaction, TreeMolecule
import yaml
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'parrot'))
from packages.rcr_torch_version.baseline_condition_model import NeuralNetContextRecommender as RCRAPI
from packages.yield_predictor.yield_predict import YieldPredictorAPI
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return ''
    
def merge_reaction_rcr(rxn, s, r, c):
    react, product = rxn.split('>>')
    for x in [s, r, c]:
        if x != '':
            react += f'.{x}'
    precursor = canonicalize_smiles(react)
    return f'{precursor}>>{product}'

def merge_reaction_parrot(rxn, c1, s1, s2, r1, r2):
    react, product = rxn.split('>>')
    for x in [c1, s1, s2, r1, r2]:
        if x != '':
            react += f'.{x}'
    precursor = canonicalize_smiles(react)
    return f'{precursor}>>{product}'

def retro_reaction_to_forward_reaction(reaction: TemplatedRetroReaction):
    try:
        retro_reaction_smiles = reaction.reaction_smiles()
        product, reactant = retro_reaction_smiles.split('>>')
        return f'{reactant}>>{product}'
    except:
        return np.nan

class RCRYieldPredictorAPI:
    def __init__(self, use_parrot=False, cuda_device=-1) -> None:
        self.use_parrot = use_parrot
        self.package_path = os.path.dirname(__file__)
        self.yield_model_path = os.path.join(self.package_path, 'yield_predictor/model')
        if not use_parrot:
            self.condition_predictor_path = os.path.join(self.package_path, 'rcr_torch_version/model')

            self.condition_predictor = RCRAPI()   # 现在只能用cpu运行
            self.condition_predictor.load_nn_model(
                info_path=self.condition_predictor_path ,
                weights_path=os.path.join(self.condition_predictor_path, 'dict_weights.npy')
                )
        else:
            self.condition_predictor_path = os.path.join(self.package_path, 'parrot')
            config = yaml.load(open(os.path.join(self.condition_predictor_path, 'configs', 'config_inference_use_uspto.yaml'), "r"),
                       Loader=yaml.FullLoader)
            
            print(
                '\n########################\nParrot configs:\n########################\n'
            )
            print(yaml.dump(config))
            print('########################\n')
            model_args = config['model_args']
            self.model_args = model_args
            self.inference_args = config['inference_args']
            dataset_args = config['dataset_args']
            try:
                model_args['use_temperature'] = dataset_args['use_temperature']
                print('Using Temperature:', model_args['use_temperature'])
            except:
                print('No temperature information is specified!')
            condition_label_mapping = inference_load(**dataset_args)
            model_args['decoder_args'].update({
                'tgt_vocab_size':
                len(condition_label_mapping[0]),
                'condition_label_mapping':
                condition_label_mapping
            })

            trained_path = model_args['best_model_dir']
            trained_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'parrot', trained_path))
            self.condition_predictor = ParrotConditionPredictionModel(
                "bert",
                trained_path,
                args=model_args,
                use_cuda=False,
                cuda_device=cuda_device)
            
        self.predict_n = 10
        self.yield_predictor = YieldPredictorAPI(model_state_path=self.yield_model_path, cuda_device=cuda_device)
        
        
    def _filtration_rcr_condition_results(self, rxn_smiles_list, condition_results):
        
        combination_df = pd.DataFrame()
        for idx, (context_combos, context_combo_scores) in enumerate(condition_results):
            condition_df = pd.DataFrame(context_combos)
            condition_df.columns = ['Temperature', 'Solvent', 'Reagent', 'Catalyst', 'null1', 'null2']
            condition_df['Score'] = context_combo_scores
            condition_df = condition_df[['Solvent', 'Reagent', 'Catalyst', 'Score']]
            for col in ['Solvent', 'Reagent', 'Catalyst']:
                condition_df[col] = condition_df[col].apply(lambda x:canonicalize_smiles(x))
            
            condition_df['react>>prod'] =  [rxn_smiles_list[idx]] * condition_df.shape[0]
            condition_df['local_rxn_id'] = [idx] * condition_df.shape[0]
            condition_df['rxn_smiles'] = condition_df.apply(lambda x:merge_reaction_rcr(
                x['react>>prod'],
                x['Solvent'],
                x['Reagent'],
                x['Catalyst']), axis=1)
            combination_df = pd.concat([combination_df, condition_df], axis=0)
        return combination_df
    
    def _filtration_parrot_condition_results(self, rxn_smiles_list, condition_results):
        
        combination_df = pd.DataFrame()
        for idx, one_pred in enumerate(condition_results):
            conditions, scores = zip(*one_pred[:self.predict_n])
            condition_df = pd.DataFrame(conditions)
            condition_df.columns = [
                'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2'
            ]
            condition_df['Score'] = (np.array(scores) / np.array(scores).sum()).tolist()
            condition_df['react>>prod'] = [rxn_smiles_list[idx]] * condition_df.shape[0]
            condition_df['local_rxn_id'] = [idx] * condition_df.shape[0]
            condition_df = condition_df[[
                'react>>prod', 'local_rxn_id', 'catalyst1', 'solvent1', 'solvent2', 'reagent1',
                'reagent2', 'Score'
            ]]
            condition_df['rxn_smiles'] = condition_df.apply(lambda x:merge_reaction_parrot(
                x['react>>prod'],
                x['catalyst1'],
                x['solvent1'],
                x['solvent2'],
                x['reagent1'],
                x['reagent2']), axis=1)

            combination_df = pd.concat([combination_df, condition_df], axis=0)
        return combination_df
    
    def predict(self, rxn_smiles_list, to_yield_class=False):        # 无反应条件的SMILES
        if not self.use_parrot:
            condition_results = [self.condition_predictor.get_n_conditions(x, self.predict_n, return_scores=True) for x in rxn_smiles_list]
            combination_df = self._filtration_rcr_condition_results(rxn_smiles_list, condition_results)
        else:
            condition_input_df = pd.DataFrame({
                'text': rxn_smiles_list,
                'labels': [[0] * 7] * len(rxn_smiles_list) if not self.model_args['use_temperature'] else [[0] * 8] * len(rxn_smiles_list) 
            })
            pred_conditions, _ = self.condition_predictor.condition_beam_search(
                condition_input_df,
                output_dir=self.model_args['best_model_dir'],
                beam=self.inference_args['beam'],
                test_batch_size=8,
                calculate_topk_accuracy=False)
            condition_results = pred_conditions
            combination_df = self._filtration_parrot_condition_results(rxn_smiles_list, condition_results)
            
            

        combination_df['Yield'] = self.yield_predictor.predict(combination_df['rxn_smiles'].tolist(), to_yield_class=to_yield_class)
        return combination_df
    
    def score_single_step_with_yield(self, aizyth_actions: List[TemplatedRetroReaction], aizyth_priors: List[float], yield_score_parameter: float = 1.2):
        
        products = [x.mol.smiles for x in aizyth_actions]
        rxn_smiles_list = [retro_reaction_to_forward_reaction(x) for x in aizyth_actions]
        
        valid_rxn_df = pd.DataFrame(
            {
                'products': products,
                'possible_actions': aizyth_actions,
                'rxn_smiles': rxn_smiles_list,
                'priors': aizyth_priors,
            }
        )
        valid_rxn_df = valid_rxn_df.dropna(subset=['rxn_smiles']).reset_index(drop=True)
        
        rxn_smiles_list = valid_rxn_df['rxn_smiles'].tolist()
        if rxn_smiles_list:
            results = self.predict(valid_rxn_df['rxn_smiles'].tolist(), to_yield_class=True)
            results['Yield_based_score'] = (4-results['Yield']) * results['Score']
            valid_rxn_df['Yield_based_score'] = results['Yield_based_score'].groupby(results['local_rxn_id']).mean()

            valid_rxn_df['new_priors'] = valid_rxn_df['priors'] + valid_rxn_df['Yield_based_score'] * yield_score_parameter
            group_sum_dict = valid_rxn_df['new_priors'].groupby(valid_rxn_df['products']).sum().to_dict()
            valid_rxn_df['priors_sum'] = valid_rxn_df['products'].apply(lambda x:group_sum_dict[x])
            valid_rxn_df['new_priors'] = valid_rxn_df['new_priors'] / valid_rxn_df['priors_sum']
            valid_rxn_df = valid_rxn_df.sort_values(by=['new_priors'], ascending=False).reset_index(drop=True)
            return valid_rxn_df['possible_actions'].tolist(), valid_rxn_df['new_priors'].tolist()

        else:
            return [], []

        
if __name__ == '__main__':

    predictor = RCRYieldPredictorAPI(use_parrot=True, cuda_device=-1)
    
    # print(predictor.predict(['CC1(C)OBOC1(C)C.Cc1ccc(Br)cc1>>Cc1cccc(B2OC(C)(C)C(C)(C)O2)c1', 'CC1(C)OBOC1(C)C.Cc1ccc(Br)cc1>>Cc1cccc(B2OC(C)(C)C(C)(C)O2)c1'], to_yield_class=True))
    
    mol1 = TreeMolecule(parent=None, smiles='CN1CCC(c2c[nH]c3ccc(NC(=O)c4ccc(F)cc4F)nc23)CC1')
    mol2 = TreeMolecule(parent=None, smiles='CN1CC(c2c[nH]c3ccc(NC(=O)c4ccc(F)cc4F)nc23)CC1')
    test_actions = [TemplatedRetroReaction(mol1, smarts='[#7;a:4]:[c:5]-[NH;D2;+0:6]-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3]>>Cl-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3].[#7;a:4]:[c:5]-[NH2;D1;+0:6]', use_rdchirl=True)] * 3 + [TemplatedRetroReaction(mol2, smarts='[#7;a:4]:[c:5]-[NH;D2;+0:6]-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3]>>Cl-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3].[#7;a:4]:[c:5]-[NH2;D1;+0:6]', use_rdchirl=True), TemplatedRetroReaction(mol2, smarts='[#7;a:4]:[c:5]-[NH;D2;+0:6]-[c:3]>>Cl', use_rdchirl=True)]
    
    # test_actions = [TemplatedRetroReaction(mol2, smarts='[#7;a:4]:[c:5]-[NH;D2;+0:6]-[c:3]>>Cl', use_rdchirl=True)] * 5
    
    test_priors = [0.5] * 3 + [0.9, 0.3]
    test_priors = (np.array(test_priors) / sum(test_priors) ).tolist()
    print(predictor.score_single_step_with_yield(test_actions, test_priors, yield_score_parameter=1.2))
    
    
    
    
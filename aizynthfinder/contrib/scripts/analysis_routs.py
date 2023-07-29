import pandas as pd
from queue import Queue
import json


def tree_to_rxn_str(dict_tree: dict):
    rxns = []
    rxn_tree_queue = Queue()
    deep_cnt = 0
    rxn_tree_queue.put((dict_tree, deep_cnt))
    while not rxn_tree_queue.empty():
        mol_node, deep_cnt = rxn_tree_queue.get()
        assert mol_node['type'] == 'mol'
        if 'children' not in mol_node: continue
        assert len(mol_node['children']) == 1
        reaction_node = mol_node['children'][0]
        reactants = []
        deep_cnt += 1
        for c_mol_node in reaction_node['children']:
            reactants.append(c_mol_node['smiles'])
            rxn_tree_queue.put((c_mol_node, deep_cnt))
        reactants.sort()
        reactants = '.'.join(reactants)
        rxn_smiles = '{}>>{}'.format(reactants, mol_node['smiles'])
        rxns.append(f'{deep_cnt} {rxn_smiles}')
    rxns.sort()
    rxn_str = ' | '.join(rxns)
        
    
    return rxn_str


if __name__ == '__main__':
    
    outputs_AZ = pd.read_json('/mnt/d/work/yield-score-analysis/aizynthfinder/aizynthfinder/interfaces/outputs_50_AZ.json.gz', 'table')
    outputs_AZ_egret = pd.read_json('/mnt/d/work/yield-score-analysis/aizynthfinder/aizynthfinder/interfaces/outputs_50_AZ_Egret.json.gz', 'table')
    
    trees_AZ = outputs_AZ.trees.tolist()
    tree_to_rxn_str(trees_AZ[0][0])   
    
    trees_AZ_egret = outputs_AZ_egret.trees.tolist()
    tree_to_rxn_str(trees_AZ_egret[0][0])  

    print()
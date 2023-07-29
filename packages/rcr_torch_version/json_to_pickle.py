import json
import pickle




if __name__ == '__main__':
    json_path = '/home/xiaoruiwang/data/ubuntu_work_beta/multi_step_work/yield_score_analysis/packages/rcr_torch_version/model/s2_dict.json'
    pickle_path = '/home/xiaoruiwang/data/ubuntu_work_beta/multi_step_work/yield_score_analysis/packages/rcr_torch_version/model/s2_dict.pickle'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data_from_json = json.load(f)
    
    data = {int(k): v for k,v in data_from_json.items()}
        
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
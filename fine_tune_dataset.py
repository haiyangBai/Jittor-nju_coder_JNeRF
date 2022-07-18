import os
import numpy as np
import json
from pprint import pprint


def get_fine_tune_data(root, exp_name, fine_tune_path, top_k):
    """
    Get fine tune images whose camera position is near enough.

    params:
        root, exp_name: data file path and data name
        top_k: [list] chose the k nearest images to test views. 
    """
    data_dir = os.path.join(root, exp_name)
    fine_tune_path = os.path.join(root, fine_tune_path, exp_name)
    train_mat = []
    test_mat = []

    with open(os.path.join(data_dir, 'transforms_train.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, 'transforms_val.json'), 'r') as g:
        val_data = json.load(g)
    train_data['frames'] += val_data['frames']

    for path in train_data['frames']:
        train_mat.append(np.reshape(path['transform_matrix'], (-1, 1)).squeeze(1))

    with open(os.path.join(data_dir, 'transforms_test.json'), 'r') as f:
        test_data = json.load(f)
    for path in test_data['frames']:
        test_mat.append(np.reshape(path['transform_matrix'], (-1, 1)).squeeze(1))

    nearest_id = []
    for mat in test_mat:
        res = np.argsort(np.mean(abs(mat - train_mat), 1))
        nearest_id += list(res[:10])
    length = len(train_data['frames'])

    i = 0
    for id in range(length):
        if id not in nearest_id:
            train_data['frames'].pop(id - i)
            i += 1
    with open(os.path.join(fine_tune_path, 'transforms_train.json'), 'w') as f:
        js = json.dumps(train_data, sort_keys=True, indent=4, separators=(',', ':'))
        f.write(js)

if __name__ == '__main__':
    root = './data/nerf_synthetic_data'
    fine_tune_path = './data/fine_tune'
    for exp_name in ['Car', 'Coffee', 'Easyship', 'Scar', 'Scarf']:
        get_fine_tune_data(root, exp_name, fine_tune_path, 10)

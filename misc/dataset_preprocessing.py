# -*- coding:utf-8 -*-
# @Desc: generate train/val splits based on metadata
import os
import os.path as osp
from pathlib import Path
import pickle
import random
import numpy as np
import argparse


import json


def main(x2c_path):
    all_data = []
    with open(osp.join(x2c_path, "metadata.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    random.shuffle(all_data)
    num_train = int(len(all_data) * 0.8)

    train_data = {'img_path': [], 'ctrl_values': []}
    val_data = {'img_path': [], 'ctrl_values': []}
    for i, item in enumerate(all_data):
        ctrl_value = np.array(item['ctrl_value'], dtype=np.float32)
        filename = item['file_name']
        img_idx, _ = osp.splitext(filename)
        folder_idx = int(img_idx) // 10000
        images_folder = f'images{folder_idx}'
        filename = osp.join(x2c_path, images_folder, filename)
        if not osp.exists(filename):
            raise ValueError(f'penny stops here!!! [{filename}]')
        if i < num_train:
            train_data['img_path'].append(filename)
            train_data['ctrl_values'].append(ctrl_value)
        else:
            val_data['img_path'].append(filename)
            val_data['ctrl_values'].append(ctrl_value)

    with open(osp.join(x2c_path, 'train_split.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(osp.join(x2c_path, 'val_split.pkl'), 'wb') as f:
        pickle.dump(val_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2c", type=str, required=True, help="Path to X2C dataet")
    args = parser.parse_args()
    main(args.x2c)
   
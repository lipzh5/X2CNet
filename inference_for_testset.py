# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle
from models.mapping_network import X2Control
from datasets import ImageProcessor
import os
import os.path as osp
from const import *
from losses import LnLosswNorm, LnLossParts


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	random.seed(cfg.seed)
	device_id = cfg.device_id
	model = X2Control(cfg).cuda(device_id)

	# saved_model_name = cfg.train.save_model_name if cfg.train.save_model_name else 'x2control.pth'
	state_dict = torch.load(osp.join(cfg.train.save_model_path, cfg.train.save_model_name),
	                        map_location=torch.device(f'cuda:{device_id}'))
	model.load_state_dict(state_dict['model'])
	model.eval()
	model.requires_grad_(False)

	image_processor = ImageProcessor(cfg.data, split_type='val')

	animated_test_images_root = '../liveportrait/animated_images_testset'
	test_split_path = '../emotionimitation/datasets/real_test_split_filtered.pkl'
	# test_split_path = '../emotionimitation/datasets/real_test_split_filtered.pkl'
	# NOTE: 1. get animated image paths
	with open(test_split_path, 'rb') as file:
		loaded_data = pickle.load(file)
	print(f'type of loaded data:{type(loaded_data)}, {loaded_data.keys()}')
	img_paths = loaded_data['img_paths']
	gt_control_vals = loaded_data['control_vals']

	masks = []
	for ctrl_val in gt_control_vals:
		_mask = np.ones(len(ctrl_val))
		if ctrl_val[6] < 0 and ctrl_val[7] < 0:
			_mask[8] = 0
			_mask[9] = 0
		masks.append(_mask)
	masks = torch.tensor(np.array(masks)).cuda(device_id)

	animated_img_paths = []
	for img_path in img_paths:
		img_name, ext = osp.splitext(osp.basename(img_path))
		folder_name = osp.basename(osp.dirname(img_path))
		animated_img_name = f'ameca_neutral--{img_name}.jpg'
		animated_img_path = osp.join(animated_test_images_root, folder_name, animated_img_name)
		animated_img_paths.append(animated_img_path)

	processed_imgs = [image_processor(img_path) for img_path in animated_img_paths]
	processed_imgs = torch.stack(processed_imgs).cuda(device_id)  # [bs, 3, 224, 224]

	pos_pred_vals = model(processed_imgs)
	print(f'total predicted len: {pos_pred_vals.shape}, total ordered controls: {len(ORDERED_CTRLS)}')
	print(f'type of pos pred vals: {type(pos_pred_vals)}')

	print(f'metric: {type(cfg.eval.metric)}')

	eval_metric = cfg.eval.metric
	if cfg.eval.normalize:
		loss_fn = LnLosswNorm(loss_fn=eval_metric)
	elif cfg.eval.sep_parts:
		loss_fn = LnLossParts(loss_fn=eval_metric)
	else:
		loss_fn = nn.L1Loss() if eval_metric == 'l1' else nn.MSELoss()
	loss_fn = loss_fn.cuda(device_id)

	gt_control_vals = torch.tensor(np.array(gt_control_vals)).cuda(device_id)
	loss = loss_fn(pos_pred_vals * masks, gt_control_vals * masks)
	if isinstance(loss, list):
		print(f'losses are: {[t.cpu().item() for t in loss]}')
	else:
		print(f'loss is: {loss} \n ******')



if __name__ == "__main__":
	# visualize_embd()
	main()
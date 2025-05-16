# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import os.path as osp
import torch
from const import *


from losses import FeatureMatchingLoss, FeatureMatchingLossL1


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1, accu_steps=1):
		self.val = val
		self.sum += val * n * accu_steps
		self.count += n
		self.avg = self.sum / self.count


def save_model(model, optimizer, cfg, trial_name, model_name='x2control.pth'):
	save_path = osp.join(cfg.train.save_model_path, trial_name)
	if not osp.exists(save_path):
		os.makedirs(save_path)

	save_path = osp.join(save_path, model_name)
	state = {
		'opt': cfg,
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
	}
	torch.save(state, save_path)
	print(f'saved model at: {save_path} \n******')


def get_delta_from_neutral(ordered_ctrls):
	delta = []
	for i, ctrl in enumerate(ORDERED_CTRLS):
		delta.append(ordered_ctrls[:, i] - FACE_NEUTRAL[ctrl])
	return torch.stack(delta, dim=1)



def get_delta_from_neutral_hlv(ordered_ctrls):
	delta = []
	for i, ctrl in enumerate(ORDERED_CTRLS_HIGH_LEVEL):
		delta.append(ordered_ctrls[:, i] - FACE_NEUTRAL_HLV[ctrl])
	return torch.stack(delta, dim=1)


def prefix(filename):
	"""a.jpg -> a"""
	pos = filename.rfind(".")
	if pos == -1:
		return filename
	return filename[:pos]


def basename(filename):
	"""a/b/c.jpg -> c"""
	return prefix(osp.basename(filename))


def get_feature_matching_loss_fn(fn_code=1):
	if fn_code == 1:
		return FeatureMatchingLoss()
	if fn_code == 2:
		return FeatureMatchingLossL1()
	if fn_code == 3:
		return FeatureMatchingLossL1(criterion='l2')
	return None



def normalize_value(value, original_min, original_max, target_min, target_max):
	"""
	Normalize a value from an original range [original_min, original_max] to a target range [target_min, target_max].

	Args:
	value (float): The value to be normalized.
	original_min (float): The minimum of the original range.
	original_max (float): The maximum of the original range.
	target_min (float): The minimum of the target range.
	target_max (float): The maximum of the target range.

	Returns:
	float: The normalized value in the target range.
	"""
	# Normalize the value to [0, 1] first
	# print(f'value original: {value[:10]}, \n after: {(value-original_min)[:10]}')
	normalized_value = (value - original_min) / (original_max - original_min)

	# Scale it to the target range [target_min, target_max]
	normalized_value_in_target_range = target_min + normalized_value * (target_max - target_min)

	return normalized_value_in_target_range
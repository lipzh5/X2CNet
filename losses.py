# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import torch
import torch.nn as nn
import torch.nn.functional as F
from const import ORDERED_CTRLS_RANGE


class FeatureMatchingLossL1(nn.Module):
	r"""Compute feature matching loss"""

	def __init__(self, criterion='l1'):
		super(FeatureMatchingLossL1, self).__init__()
		if criterion == 'l1':
			self.criterion = nn.L1Loss()
		elif criterion == 'l2' or criterion == 'mse':
			self.criterion = nn.MSELoss()
		else:
			raise ValueError('Criterion %s is not recognized' % criterion)

	def forward(self, pred_features, target_features):
		r"""Return the target vector for the binary cross entropy loss
		computation.
		Args:
		   fake_features (list of lists): Discriminator features of fake images.
		   real_features (list of lists): Discriminator features of real images.

		Returns:
		   (tensor): Loss value.
		"""
		loss = self.criterion(pred_features, target_features)

		return loss


class FeatureMatchingLoss(nn.Module):

	def forward(self, pred_features, target_features):
		cos_sim = F.cosine_similarity(pred_features, target_features)
		cos_sim = torch.mean(cos_sim)
		return 1 - cos_sim



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


class LnLosswNorm(nn.Module):
	def __init__(self, loss_fn='l1'):
		super().__init__()
		if loss_fn == 'l1':
			self.criterion = nn.L1Loss()
		elif loss_fn == 'l2':
			self.criterion = nn.MSELoss()

	def forward(self, predicts, targets):  # (bs, 30)
		# normalization
		predicts_normalized = []
		targets_normalized = []
		for idx, item in enumerate(ORDERED_CTRLS_RANGE):
			[original_min, original_max] = item[1]

			cur_pred = predicts[:, idx]
			cur_target = targets[:, idx]

			cur_pred_normalized = normalize_value(cur_pred, original_min, original_max, -1, 1)
			cur_target_normalized = normalize_value(cur_target, original_min, original_max, -1, 1)
			predicts_normalized.append(cur_pred_normalized)
			targets_normalized.append(cur_target_normalized)

		predicts_normalized = torch.stack(predicts_normalized, dim=1)
		targets_normalized = torch.stack(targets_normalized, dim=1)
		print(torch.mean(abs(predicts_normalized - targets_normalized), dim=0))
		return self.criterion(predicts_normalized, targets_normalized)


class LnLossParts(nn.Module):
	def __init__(self, loss_fn='l1'):
		super().__init__()
		if loss_fn == 'l1':
			self.criterion = nn.L1Loss()
		elif loss_fn == 'l2':
			self.criterion = nn.MSELoss()

	def forward(self, predicts, targets):  # (bs, 30)
		# Brow: 0-3; Eyelids: 4-7; Gaze: 8-9; neck/head: 10-12 + 27-28; lip: 13-26; nose: 29
		brow_predicts, brow_targets = predicts[:, 0:4], targets[:, 0:4]
		eyelids_predicts, eyelids_targets = predicts[:, 4:8], targets[:, 4:8]
		gaze_predicts, gaze_targets = predicts[:, 8:10], targets[:, 8:10]
		neck_head_predicts = torch.cat([predicts[:, 10:13], predicts[:, 27:29]], dim=1)
		neck_head_targets = torch.cat([targets[:, 10:13], targets[:, 27:29]], dim=1)
		lip_predicts, lip_targets = predicts[:, 13:27], targets[:, 13:27]
		nose_predicts, nose_targets = predicts[:, 29], targets[:, 29]
		losses = [
			self.criterion(brow_predicts, brow_targets),
			self.criterion(eyelids_predicts, eyelids_targets),
			self.criterion(gaze_predicts, gaze_targets),
			self.criterion(neck_head_predicts, neck_head_targets),
			self.criterion(lip_predicts, lip_targets),
			self.criterion(nose_predicts, nose_targets)
		]

		return losses

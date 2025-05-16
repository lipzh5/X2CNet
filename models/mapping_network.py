# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: pose regression network -- from image (and possibly language instruction) to control values
# import torchvision
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from torchvision import models

import torch.nn as nn


# TODO move to config.yaml
FEATURE_DIM = 256 * 56 * 56  # 1024 * 14 * 14
HIDDEN_DIM = 512


class X2Control(nn.Module):
	"""mapping from image to control values (corresponds to facial expression)"""

	def __init__(self, cfg):
		super().__init__()
		self.encoder_opt = cfg.model.encoder_opt
		if self.encoder_opt == 'vgg':
			self.feature_extractor = models.vgg16(pretrained=True).features
			self.pos_pred_head = nn.Sequential(
				nn.Linear(25088, 256), nn.ReLU(),
				nn.Linear(256, 256), nn.ReLU(),
				nn.Linear(256, cfg.model.pos_action_size))

		elif self.encoder_opt == 'transformer':
			model_name = "google/vit-base-patch16-224-in21k"  # Pretrained ViT
			model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=10)
			self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
			self.pos_pred_head = nn.Sequential(
				nn.Linear(151296, 256), nn.ReLU(),
				nn.Linear(256, 256), nn.ReLU(),
				nn.Linear(256, cfg.model.pos_action_size))
		else:
			# self.angle_pred_head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, cfg.model.angle_action_size))
			weights = models.ResNet18_Weights.DEFAULT if cfg.model.use_resnet_pretrain else None
			model = models.resnet18(weights=weights)  # 44.629 MB,
			self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # cls=1000

			self.pos_pred_head = nn.Sequential(
				nn.Linear(512, 256), nn.ReLU(),
				nn.Linear(256, 256), nn.ReLU(),
				nn.Linear(256, cfg.model.pos_action_size),
			)

	def extract_feature(self, x):
		output = self.feature_extractor(x)
		if self.encoder_opt == 'vgg':
			return torch.flatten(output, 1)
		elif self.encoder_opt == 'transformer':
			features = output.last_hidden_state
			return features.view(features.size(0), -1)
		elif self.encoder_opt == 'resnet':  # (bs, 512, 1, 1)
			return output.squeeze(-1).squeeze(-1)


	def forward(self, x):
		features = self.extract_feature(x)
		pos_pred_vals = self.pos_pred_head(features)
		return features, pos_pred_vals

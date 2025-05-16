# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import pickle
import os.path as osp
import os

cwd = os.getcwd()


class Transform:
	def __init__(self, cfg, split_type):  # cfg: config.data
		transform_list = [transforms.ColorJitter(**cfg.transform.color_jitter)] if (
				split_type == 'train' and cfg.apply_color_jitter) else []
		self.transform = transforms.Compose(
			transform_list +
			[transforms.ToTensor(), transforms.Normalize(mean=cfg.mean, std=cfg.std)])

	def __call__(self, data):
		return self.transform(data)


class Resize:
	def __init__(self, target_size):
		self.target_size = target_size  # cfg.resize.target_size

	def __call__(self, img_path):
		img = cv2.imread(img_path)
		interp = cv2.INTER_AREA if img.shape[0] > self.target_size else cv2.INTER_CUBIC
		resized = cv2.resize(img, dsize=(self.target_size, self.target_size), interpolation=interp)
		return resized
# return Image.fromarray(resized, mode='RGB')


class Crop2Target:
	def __init__(self, target_size):
		self.crop_op = transforms.RandomCrop(target_size)

	def __call__(self, img_path):
		img = Image.open(img_path)
		return self.crop_op(img)  # PIL.Image


class ImageProcessor:
	def __init__(self, data_cfg, split_type):
		super().__init__()
		resize_target = data_cfg.transform.resize_target
		self.resize = Resize(target_size=resize_target)
		# self.resize = Crop2Target(resize_target) if split_type == 'train' else Resize(target_size=resize_target)
		self.transform = Transform(data_cfg, split_type)
		neutral_norm_img_path = data_cfg.neutral_norm_img_path
		self.neutral_norm_img = self.resize(neutral_norm_img_path) if neutral_norm_img_path else None

	def __call__(self, img_path):
		resized = self.resize(img_path)
		resized = resized - self.neutral_norm_img if self.neutral_norm_img is not None else resized
		resized = Image.fromarray(resized, mode='RGB')
		return self.transform(resized)


class ICtrlDataset(Dataset):
	"""a collection of (img, control_values) pairs"""

	def __init__(self, cfg, split_type):
		super().__init__()
		self.cfg = cfg
		self.split_type = split_type
		self.image_processor = ImageProcessor(cfg.data, split_type)

		filename = f'{split_type}_split_cg.pkl'
		with open(osp.join(cfg.data.ictrl_data_path, filename), 'rb') as file:
			data = pickle.load(file)

		self.img_paths = data['img_path']
		self.ctrl_values = data['ctrl_values']

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		img_path = self.img_paths[index]
		img_path = osp.join('../emotionimitation', img_path)
		# penny NOTE: load animated images for feature matching
		animated_img_path = img_path.replace('cropped_frames_fine/',
		                                     'cropped_frames_fine_animated_from_vid/ameca_neutral--')

		ctrl_vals = self.ctrl_values[index]
		return self.image_processor(img_path), self.image_processor(animated_img_path), ctrl_vals

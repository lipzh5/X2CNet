# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import cv2
import os
import torch
import time
import os.path as osp
import numpy as np
from rich.progress import track
from liveportrait_configs.argument_config import ArgumentConfig
from liveportrait_configs.inference_config import InferenceConfig
from liveportrait_configs.crop_config import CropConfig
from liveportrait_utils.cropper import Cropper
from liveportrait_wrapper import LivePortraitWrapper
from liveportrait_utils.camera import get_rotation_matrix
from liveportrait_utils.io import load_image_rgb, resize_to_limit, load_video, load_images_from_bytes
from liveportrait_utils.video import get_fps
from liveportrait_utils.rprint import rlog as log
from liveportrait_utils.helper import dct2device,  calc_motion_multiplier, basename
from liveportrait_utils.video import images2video
from PIL import Image


class X2CNetPipeline:
	def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
		self.wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
		self.cropper = Cropper(crop_cfg=crop_cfg)
		self.init_source(inference_cfg)


	def init_source(self, inf_cfg: InferenceConfig):
		'''source features are fixed in our application'''
		img_rgb = load_image_rgb(inf_cfg.source)
		img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
		log(f"Load source image from {inf_cfg.source}")
		source_lmk = self.cropper.calc_lmk_from_cropped_image(img_rgb)
		img_crop_256x256 = cv2.resize(img_rgb, (256, 256))  # force to resize to 256x256
		I_s = self.wrapper.prepare_source(img_crop_256x256)
		x_s_info = self.wrapper.get_kp_info(I_s)
		self.x_c_s = x_s_info['kp']
		self.R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
		self.x_s_info = x_s_info
		self.source_kp = self.wrapper.transform_keypoint(x_s_info)
		self.source_feat = self.wrapper.extract_feature_3d(I_s)
		
	
	def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
		n_frames = I_lst.shape[0]
		template_dct = {
			'n_frames': n_frames,
			'output_fps': kwargs.get('output_fps', 25),
			'motion': [],
			'c_eyes_lst': [],
			'c_lip_lst': [],
		}

		for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
			# collect s, R, δ and t for inference
			I_i = I_lst[i]
			x_i_info = self.wrapper.get_kp_info(I_i)
			x_s = self.wrapper.transform_keypoint(x_i_info)
			R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

			item_dct = {
				'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
				'R': R_i.cpu().numpy().astype(np.float32),
				'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
				't': x_i_info['t'].cpu().numpy().astype(np.float32),
				'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
				'x_s': x_s.cpu().numpy().astype(np.float32),
			}

			template_dct['motion'].append(item_dct)

			c_eyes = c_eyes_lst[i].astype(np.float32)
			template_dct['c_eyes_lst'].append(c_eyes)

			c_lip = c_lip_lst[i].astype(np.float32)
			template_dct['c_lip_lst'].append(c_lip)

		return template_dct

	def execute(self, args: ArgumentConfig):
		device = self.wrapper.device
		
		ts = time.time()
		output_fps = int(get_fps(args.driving))
		log(f"Load driving video from: {args.driving}, FPS is {output_fps}")
		driving_rgb_lst = load_video(args.driving)
		# driving_rgb_lst = load_image_sequences(args.driving_image_folder, max_len=400)
		# driving_rgb_lst = load_images_from_bytes(self.frame_buffer)  # TODO debug
		print(f'load image sequence time: {time.time() - ts}')

		ts = time.time()
		n_frames = len(driving_rgb_lst)
		ret_d = self.cropper.crop_driving_video(driving_rgb_lst)
		driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']

		c_d_eyes_lst, c_d_lip_lst = self.wrapper.calc_ratio(driving_lmk_crop_lst)
		n_frames = min(n_frames,  len(driving_rgb_crop_lst))
		driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
		I_d_lst = self.wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
		# Penny TODO make the first motion to be neutral
		driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)
		print(f'make motion template time: {time.time() - ts}')
		I_p_lst = []
		ctrl_values_all = []
		ts = time.time()
		for i in track(range(n_frames), description='🚀Animating...', total=n_frames):
			x_d_i_info = driving_template_dct['motion'][i]
			x_d_i_info = dct2device(x_d_i_info, device)
			R_d_i = x_d_i_info['R']
			if i == 0:  # cache the first frame
				R_d_0 = R_d_i
				x_d_0_info = x_d_i_info.copy()

			delta_new = self.x_s_info['exp'].clone()
			# if inf_cfg.flag_relative_motion:
			R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ self.R_s
			delta_new = self.x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
			scale_new = self.x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
			t_new = self.x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
			t_new[..., 2].fill_(0)  # zero tz
			x_d_i_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new
			if i == 0:
				x_d_0_new = x_d_i_new
				motion_multiplier = calc_motion_multiplier(self.source_kp, x_d_0_new)
					# motion_multiplier *= inf_cfg.driving_multiplier
			x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
			x_d_i_new = x_d_diff + self.source_kp
			# x_d_i_new = self.wrapper.stitching(self.source_kp, x_d_i_new)
			# x_d_i_new = self.source_kp + (x_d_i_new - self.source_kp) * inf_cfg.driving_multiplier
			out = self.wrapper.warp_decode(self.source_feat, self.source_kp, x_d_i_new)
			ctrl_values = self.wrapper.get_control_values(out['out'])[0]
			ctrl_values_all.append(ctrl_values.cpu())

			if args.export_video:
				I_p_i = self.wrapper.parse_output(out['out'])[0]
				I_p_lst.append(I_p_i)
		
		if args.export_video:
			wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.mp4')
			images2video(I_p_lst, wfp=wfp, fps=output_fps)
		return ctrl_values_all

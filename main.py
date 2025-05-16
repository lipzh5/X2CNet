# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import os
import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
import random
import time
from torch.utils.data import DataLoader
import transformers
from models.mapping_network import X2Control
from datasets import ICtrlDataset
from torch.utils.tensorboard import SummaryWriter
from utils import *
# from losses import FeatureMatchingLoss
import logging

log = logging.getLogger(__name__)


def train(cfg, train_loader, model, optimizer, scheduler, loss_fn, feature_matching_loss_fn, epoch):
	losses = AverageMeter()
	model.train()
	device_id = cfg.device_id
	accu_steps = cfg.train.accumulation_steps
	optimizer.zero_grad()
	start_time = time.time()
	num_batches = len(train_loader)
	num_epochs = cfg.train.num_epochs
	angle_action_size = cfg.model.angle_action_size
	func_get_delta_vals = get_delta_from_neutral_hlv if cfg.pred_hlv_ctrls else get_delta_from_neutral
	_lambda = cfg.train.feat_matching_ratio
	for i_batch, batch in enumerate(train_loader):
		batch = [t.cuda(device_id) for t in batch]
		images, animated_images, ctrl_vals = batch
		pred_features, pos_predicts = model(images)  # [bs, action_size]

		if cfg.train.pred_delta_from_neutral:
			ctrl_vals = func_get_delta_vals(ctrl_vals)

		with torch.no_grad():
			target_features = model.feature_extractor(animated_images)
			if model.encoder_opt == 'resnet':
				target_features = target_features.squeeze(-1).squeeze(-1)
			elif model.encoder_opt == 'vgg':
				target_features = torch.flatten(target_features, 1)
			elif model.encoder_opt == 'transformer':
				target_features = target_features.last_hidden_state
				# print(f'features shape 111: {features.shape}')
				target_features = target_features.view(target_features.size(0), -1)
		feature_matching_loss = feature_matching_loss_fn(pred_features,
		                                                 target_features) if feature_matching_loss_fn is not None else 0
		# if eyelid upper left/right <0 then, mask gaze
		loss = loss_fn(pos_predicts, ctrl_vals) + _lambda * feature_matching_loss
		# print(f'total loss: {loss}, feat mat loss: {feature_matching_loss} \n -----')

		losses.update(loss.item(), ctrl_vals.shape[0])

		loss = loss / accu_steps
		loss.backward()

		if ((
			    i_batch + 1) % accu_steps) == 0:  # or (i_batch + 1) == len_train_loader: note: may drop grads of the last batch
			torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_value)
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

		if (i_batch + 1) % cfg.train.log_interval == 0:
			elapsed_time = time.time() - start_time
			log.info(
				f'**TRAIN**|Epoch {epoch}/{num_epochs} | Batch {i_batch + 1}/{num_batches} | Time/Batch(ms) {elapsed_time * 1000.0 / cfg.train.log_interval} | Train Loss {losses.avg}')
			start_time = time.time()
			print(f'total loss: {loss}, feat mat loss: {feature_matching_loss} \n -----')

	# if epoch == cfg.train.num_epochs:
	# 	print(f'predict: {pos_predicts[:, 0]}, \n ground truth: {ctrl_vals[:, 13]}')
	# 	raise ValueError('Penny stops here!!!')
	return losses.avg


def evaluate(cfg, data_loader, model, loss_fn, feature_matching_loss_fn, epoch, test=False):
	losses = AverageMeter()
	model.eval()
	num_batches = len(data_loader)
	num_epochs = cfg.train.num_epochs
	device_id = cfg.device_id
	angle_action_size = cfg.model.angle_action_size
	func_get_delta_vals = get_delta_from_neutral_hlv if cfg.pred_hlv_ctrls else get_delta_from_neutral
	_lambda = cfg.train.feat_matching_ratio
	with torch.no_grad():
		for i_batch, batch in enumerate(data_loader):
			batch = [t.cuda(device_id) for t in batch]
			images, animated_images, ctrl_vals, = batch
			pred_features, pos_predicts = model(images)
			# gt_ctrl = ctrl_vals[:, 13:27]
			if cfg.train.pred_delta_from_neutral:
				ctrl_vals = func_get_delta_vals(ctrl_vals)

			target_features = model.feature_extractor(animated_images)
			if model.encoder_opt == 'resnet':
				target_features = target_features.squeeze(-1).squeeze(-1)
			elif model.encoder_opt == 'vgg':
				target_features = torch.flatten(target_features, 1)
			elif model.encoder_opt == 'transformer':
				target_features = target_features.last_hidden_state
				# print(f'features shape 111: {features.shape}')
				target_features = target_features.view(target_features.size(0), -1)
			feature_matching_loss = feature_matching_loss_fn(pred_features,
			                                                 target_features) if feature_matching_loss_fn is not None else 0
			loss = loss_fn(pos_predicts, ctrl_vals) + _lambda * feature_matching_loss
			# print(f'feat mat loss: {feature_matching_loss}, total loss: {loss} ')
			# if epoch == cfg.train.num_epochs:
			# 	print(f'predict: {pos_predicts[:, 0]}, \n ground truth: {ctrl_vals[:, 0]}')
			# loss = loss_fn(pos_predicts , ctrl_vals[:, :-angle_action_size]) + loss_fn(angle_predicts, ctrl_vals[:, -angle_action_size:])
			losses.update(loss.item(), ctrl_vals.shape[0])
	return losses.avg


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	random.seed(cfg.seed)
	pred_delta = int(cfg.train.pred_delta_from_neutral)
	traverse_step_train = cfg.data.traverse_step_train
	traverse_step_val = cfg.data.traverse_step_val
	app_cj = int(cfg.data.apply_color_jitter)
	wp = cfg.train.warm_up
	resnet_pretrain = int(cfg.model.use_resnet_pretrain)
	neutral_norm = 1 if cfg.data.neutral_norm_img_path else 0
	anim_training = 1 if cfg.data.enable_animated_training else 0
	normalize0_1 = 1 if cfg.data.normalize0_1 else 0
	pred_hlv = 1 if cfg.pred_hlv_ctrls else 0
	trial_name = f"uda_trial_{cfg.trial}_bs{cfg.train.batch_size}_ep{cfg.train.num_epochs}_lr{cfg.train.lr}_lossfn{cfg.train.loss_fn}_matcode{cfg.train.feat_matching_func_code}_cg{cfg.data.clear_gaze}"
	_part = cfg.train.part_based
	d = {'gaze': 2, 'neck': 5, 'others': 23}
	if _part in d:
		trial_name += f'_part{_part}'
		cfg.model.pos_action_size = d[_part]

	writer = SummaryWriter(osp.join('runs', trial_name))
	log.info(f"***********\n TRIAL: {trial_name}\n STARTS!***********")
	if cfg.data.enable_animated_training:
		cfg.data.mean = [0.4708, 0.4167, 0.3200]
		cfg.data.std = [0.1411, 0.1233, 0.1123]
	# if cfg.data.enable_co_training:
	# 	cfg.data.mean = []
	# 	cfg.data.std = []
	if cfg.pred_hlv_ctrls:
		cfg.model.pos_action_size = 29

	if cfg.train.mouth_only == 1:
		cfg.model.pos_action_size = 14
	elif cfg.train.mouth_only == -1:
		cfg.model.pos_action_size = 16
	# cfg.train.save_model_name = 'x2control_hlv.pth'
	device_id = cfg.device_id
	model = X2Control(cfg).cuda(device_id)
	if cfg.train.two_stage_train == 1:  # load state dict from the 1st stage and train again
		# load stage one pretrained path
		state_dict = torch.load(
			osp.join(cfg.train.save_model_path, 'trial_0_bs32_ep30_lr0.001_animtrain1/x2control.pth'),
			map_location=torch.device(f'cuda:{device_id}'))
		model.load_state_dict(state_dict['model'])
	elif cfg.train.two_stage_train == 2:  # simulated + animated
		state_dict = torch.load(
			osp.join(cfg.train.save_model_path, 'trial_0_bs128_ep100_lr0.001_animtrain0_2stageT0/x2control.pth'),
			map_location=torch.device(f'cuda:{device_id}'))

	loss_fn = nn.L1Loss() if cfg.train.loss_fn == 1 else nn.HuberLoss(delta=0.01)
	if cfg.train.loss_fn == 3:
		loss_fn = nn.MSELoss()
	loss_fn = loss_fn.cuda(device_id)
	feature_matching_loss_fn = get_feature_matching_loss_fn(cfg.train.feat_matching_func_code)
	if feature_matching_loss_fn is not None:
		feature_matching_loss_fn = feature_matching_loss_fn.cuda(device_id)
	# feature_matching_loss_fn = FeatureMatchingLoss().cuda(device_id)

	val_set = ICtrlDataset(cfg, split_type='val')
	val_loader = DataLoader(val_set, shuffle=False, batch_size=cfg.train.batch_size * 2,
	                        num_workers=cfg.train.num_workers)
	num_epochs = cfg.train.num_epochs
	if cfg.do_eval:
		state_dict = torch.load(osp.join(cfg.train.save_model_path, cfg.train.save_model_name),
		                        map_location=torch.device(f'cuda:{device_id}'))
		model.load_state_dict(state_dict['model'])
		model.eval()
		model.requires_grad_(False)
		val_loss = evaluate(cfg, val_loader, model, loss_fn, feature_matching_loss_fn, num_epochs, test=False)
		print(f'validation loss is: {val_loss} \n **************')
		return

	train_set = ICtrlDataset(cfg, split_type='train')
	train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.train.batch_size,
	                          num_workers=cfg.train.num_workers)

	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

	'''cosine schedule with warmup'''
	total_training_steps = num_epochs * len(train_loader) // cfg.train.accumulation_steps
	scheduler = transformers.get_cosine_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=int(total_training_steps * cfg.train.warm_up),
		num_training_steps=total_training_steps)

	for epoch in range(1, num_epochs + 1):
		train_loss = train(cfg, train_loader, model, optimizer, scheduler, loss_fn, feature_matching_loss_fn, epoch)
		val_loss = evaluate(cfg, val_loader, model, loss_fn, feature_matching_loss_fn, epoch, test=False)
		log.info(f'======\n Epoch: {epoch}|Train Loss: {train_loss}| Val Loss: {val_loss} \n ======')
		# writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

		writer.add_scalar("Loss/train", train_loss, epoch)
		writer.add_scalar("Loss/val", val_loss, epoch)
	writer.close()

	# TODO optimize save model
	model_name = cfg.train.save_model_name if cfg.train.save_model_name else 'x2control.pth'
	save_model(model, optimizer, cfg, trial_name, model_name=model_name)




if __name__ == "__main__":
	main()

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


def train(cfg, train_loader, model, optimizer, scheduler, loss_fn, epoch):
	losses = AverageMeter()
	model.train()
	device_id = cfg.device_id
	accu_steps = cfg.train.accumulation_steps
	optimizer.zero_grad()
	start_time = time.time()
	num_batches = len(train_loader)
	num_epochs = cfg.train.num_epochs
	for i_batch, batch in enumerate(train_loader):
		batch = [t.cuda(device_id) for t in batch]
		images, ctrl_vals = batch
		pred_features, pos_predicts = model(images)  # [bs, action_size]
		loss = loss_fn(pos_predicts, ctrl_vals)

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
			# print(f'total loss: {loss}, feat mat loss: {feature_matching_loss} \n -----')

	# if epoch == cfg.train.num_epochs:
	# 	print(f'predict: {pos_predicts[:, 0]}, \n ground truth: {ctrl_vals[:, 13]}')
	# 	raise ValueError('Penny stops here!!!')
	return losses.avg


def evaluate(cfg, data_loader, model, loss_fn):
	losses = AverageMeter()
	model.eval()
	num_batches = len(data_loader)
	num_epochs = cfg.train.num_epochs
	device_id = cfg.device_id
	# _lambda = cfg.train.feat_matching_ratio
	with torch.no_grad():
		for i_batch, batch in enumerate(data_loader):
			batch = [t.cuda(device_id) for t in batch]
			images, ctrl_vals, = batch
			pred_features, pos_predicts = model(images)
			loss = loss_fn(pos_predicts, ctrl_vals) 
			losses.update(loss.item(), ctrl_vals.shape[0])
	return losses.avg


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	random.seed(cfg.seed)
	trial_name = f"trial_{cfg.trial}_bs{cfg.train.batch_size}_ep{cfg.train.num_epochs}_lr{cfg.train.lr}_lossfn{cfg.train.loss_fn}_matcode{cfg.train.feat_matching_func_code}"

	writer = SummaryWriter(osp.join('runs', trial_name))
	log.info(f"***********\n TRIAL: {trial_name}\n STARTS!***********")

	device_id = cfg.device_id
	model = X2Control(cfg).cuda(device_id)
	
	loss_fn = nn.L1Loss() if cfg.train.loss_fn == 1 else nn.HuberLoss(delta=0.01)
	if cfg.train.loss_fn == 3:
		loss_fn = nn.MSELoss()
	loss_fn = loss_fn.cuda(device_id)

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
		val_loss = evaluate(cfg, val_loader, model, loss_fn)
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
		train_loss = train(cfg, train_loader, model, optimizer, scheduler, loss_fn, epoch)
		val_loss = evaluate(cfg, val_loader, model, loss_fn)
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

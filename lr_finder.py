
import argparse
import glob
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from torchvision import transforms
from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader.data_loader import get_loader


from torch.optim.lr_scheduler import _LRScheduler


class FindLR(_LRScheduler):
	"""exponentially increasing learning rate

	Args:
		optimizer: optimzier(e.g. SGD)
		num_iter: totoal_iters
		max_lr: maximum  learning rate
	"""
	def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):

		self.total_iters = num_iter
		self.max_lr = max_lr
		super().__init__(optimizer, last_epoch)

	def get_lr(self):

		return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in self.base_lrs]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-net', type=str, required=True, help='net type')
	parser.add_argument('-bs', type=int, default=64, help='batch size for dataloader')
	parser.add_argument('-base_lr', type=float, default=1e-7, help='min learning rate')
	parser.add_argument('-max_lr', type=float, default=10, help='max learning rate')
	parser.add_argument('-num_iter', type=int, default=100, help='num of iteration')
	parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
	parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')
	parser.add_argument('-nclasses', type=int, default=24, help='number of classes or labels')
	parser.add_argument('-resize', type=int, default=32, help='new size to rescale image')
	parser.add_argument('-pretrained', action='store_true', default=False, help='use pretrained model or not')
	parser.add_argument('-balanced', action='store_true', default=False, help='load data with resampling to avoid training with imbalanced dataset')
	parser.add_argument('-num_workers', type=int, default=0, help='number of process for data loading')

	args = parser.parse_args()

	# cifar100_training_loader = get_training_dataloader(
	#     settings.CIFAR100_TRAIN_MEAN,
	#     settings.CIFAR100_TRAIN_STD,
	#     num_workers=4,
	#     batch_size=args.b,
	# )

	datasets, loader = get_loader(args)
	xView_train_loader = loader['training']
	xView_test_loader = loader['validation']

	net = get_network(args)
	loss_weights = torch.tensor(get_loss_weights(args)).cuda()
	loss_function = nn.CrossEntropyLoss(weight=loss_weights)
	net_params = split_weights(net)
	optimizer = optim.SGD(net_params, lr=args.base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)


	#set up warmup phase learning rate scheduler
	lr_scheduler = FindLR(optimizer, max_lr=args.max_lr, num_iter=args.num_iter)
	epoches = int(args.num_iter / len(xView_train_loader)) + 1

	n = 0

	learning_rate = []
	losses = []
	for epoch in range(epoches):

		#training procedure
		net.train()

		for batch_index, (images, labels) in enumerate(xView_train_loader):
			if n > args.num_iter:
				break

			

			images = images.cuda()
			labels = labels.cuda()

			optimizer.zero_grad()
			predicts = net(images)
			loss = loss_function(predicts, labels)
			if torch.isnan(loss).any():
				n += 1e8
				break
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			print('Iterations: {iter_num} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.8f}'.format(
				loss.item(),
				optimizer.param_groups[0]['lr'],
				iter_num=n,
				trained_samples=batch_index * args.bs + len(images),
				total_samples=len(xView_train_loader.dataset),
			))

			learning_rate.append(optimizer.param_groups[0]['lr'])
			losses.append(loss.item())
			n += 1

	learning_rate = learning_rate[10:-5]
	losses = losses[10:-5]

	fig, ax = plt.subplots(1,1)
	ax.plot(learning_rate, losses)
	ax.set_xlabel('learning rate')
	ax.set_ylabel('losses')
	ax.set_xscale('log')
	ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

	fig.savefig('result.jpg')

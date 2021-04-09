""" Train PyTorch model.

Author: Natnael Daba

"""
import sys
import argparse
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data_loader.data_loader import get_loader
from config import *
from utils import *


def train(epoch):

	start = time.time()
	net.train()

	for batch_index, (images, labels) in enumerate(xView_train_loader):

		if args.gpu:
			labels = labels.cuda()
			images = images.cuda()

		optimizer.zero_grad()
		outputs = net(images)
		loss = loss_function(outputs, labels)
		loss.backward()
		optimizer.step()
		

		n_iter = (epoch - 1) * len(xView_train_loader) + batch_index + 1

		last_layer = list(net.children())[-1]
		for name, para in last_layer.named_parameters():
			if 'weight' in name:
				writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
			if 'bias' in name:
				writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

		print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
		loss.item(),
		optimizer.param_groups[0]['lr'],
		epoch=epoch,
		trained_samples=batch_index * args.bs + len(images),
		total_samples=len(xView_train_loader.dataset)))

		#update training loss for each iteration
		writer.add_scalar('Train/loss', loss.item(), n_iter)

		if epoch <= args.warm:
			warmup_scheduler.step()

	for name, param in net.named_parameters():
		layer, attr = os.path.splitext(name)
		attr = attr[1:]
		writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

	finish = time.time()

	print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

	start = time.time()
	net.eval()

	test_loss = 0.0 
	correct = 0.0

	all_predictions = []
	all_targets = []
	for (images, labels) in xView_test_loader:

		if args.gpu:
			images = images.cuda()
			labels = labels.cuda()

		outputs = net(images)
		loss = loss_function(outputs, labels)

		test_loss += loss.item() * images.size(0)
		_, preds = outputs.max(1)
		correct += preds.eq(labels).sum()

		all_predictions.extend(preds.cpu().tolist())
		all_targets.extend(labels.cpu().tolist())

	finish = time.time()
	
	matrix = confusion_matrix(all_targets, all_predictions)
	fig = plot_confusion_matrix(matrix, class_names, normalize=True)
	fig.savefig(os.path.join(save_plot_directory,'confusion_matrix_epoch_'+str(epoch)+'.png'), bbox_inches='tight')
	save_report(all_targets, all_predictions, class_names, reports_directory, epoch)

	writer.add_figure('Test/Confusion Matrix', fig, epoch)

	if args.gpu:
		print('GPU INFO.....')
		print(torch.cuda.memory_summary(), end='')
	print('Evaluating Network.....')
	average_loss = test_loss / len(xView_test_loader.dataset)
	accuracy = correct.float() / len(xView_test_loader.dataset)
	print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
		epoch,
		average_loss,
		accuracy,
		finish - start
	))
	print()
	
	#add information to tensorboard
	if tb:
		writer.add_scalar('Test/Average loss', average_loss, epoch)
		writer.add_scalar('Test/Accuracy', accuracy, epoch)

	return correct.float() / len(xView_test_loader.dataset)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-net', type=str, required=True, help='net type')
	parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
	parser.add_argument('-bs', type=int, default=128, help='batch size for dataloader')
	parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
	parser.add_argument('-nclasses', type=int, default=24, help='number of classes or labels')
	parser.add_argument('-weight_decay', type=float, default=1e-3, help='weight decay to be used by optimizer')
	parser.add_argument('-resize', type=int, default=32, help='new size to rescale image')
	parser.add_argument('-pretrained', action='store_true', default=False, help='use pretrained model or not')
	parser.add_argument('-resume', action='store_true', default=False, help='resume training')
	parser.add_argument('-balanced', action='store_true', default=False, help='load data with resampling to avoid training with imbalanced dataset')
	parser.add_argument('-gamma', type=float, default=0.1, help='learning rate decay factor')
	parser.add_argument('-step_size', type=int, default=20, help='number of epochs to wait before decaying learning rate')
	parser.add_argument('-momentum', type=int, default=0.9, help='momentum for optimizer')
	parser.add_argument('-epochs', type=int, default=1000, help='number of epochs to train for')
	parser.add_argument('-weighted_loss', action='store_true', default=False, help='weight the loss according to class distribution')
	parser.add_argument('-num_workers', type=int, default=0, help='number of process for data loading')
	parser.add_argument('-warm', type=int, default=1, help='first number of batches to use for warm up')
	parser.add_argument('-no_bias_decay', action='store_true', default=False, help='If True, L2 weight decay won\'t be applied to conv and linear bias')

	args = parser.parse_args()

	print(vars(args))

	datasets, loader = get_loader(args)
	#print("len datasetse", len(datasets['training'].classes))
	if args.nclasses != len(datasets['training'].classes):
		print('There is a mismatch between specified number of classes and actual number of classes in the dataset.')
		quit()

	xView_train_loader = loader['training']
	xView_test_loader = loader['validation']
	#first_batch = next(iter(xView_train_loader)) # For sanity check by overfitting a mini batch

	net = get_network(args)
	#net.apply(init_weights)
	net = init_weights2(net)
	if args.weighted_loss:
		loss_weights = torch.tensor(get_loss_weights(args)).cuda()
	else:
		loss_weights = None
	
	if args.no_bias_decay:
		net_params = split_weights(net)
		optimizer = optim.SGD(net_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	else:
		optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
		
	loss_function = nn.CrossEntropyLoss(weight=loss_weights) 
	
	iter_per_epoch = len(xView_train_loader)
	warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

	#use tensorboard
	
	if not os.path.exists(LOG_DIR):
		os.mkdir(LOG_DIR)
	
	save_plot_directory = os.path.join(PLOTS_DIRECTORY, args.net+'/', TIME_NOW+'/')

	reports_directory = os.path.join(REPORTS_DIRECTORY, args.net+'/', TIME_NOW+'/')

	if not os.path.exists(os.path.join(PLOTS_DIRECTORY, args.net+'/')):
		os.mkdir(os.path.join(PLOTS_DIRECTORY, args.net+'/'))

	if not os.path.exists(save_plot_directory):
		os.mkdir(save_plot_directory)

	if not os.path.exists(os.path.join(REPORTS_DIRECTORY, args.net+'/')):
		os.mkdir(os.path.join(REPORTS_DIRECTORY, args.net+'/'))

	if not os.path.exists(reports_directory):
		os.mkdir(reports_directory)

	if args.resume:
		recent_folder = most_recent_folder(os.path.join(CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
		if not recent_folder:
			raise Exception('no recent folder were found')

		checkpoint_path = os.path.join(CHECKPOINT_PATH, args.net, recent_folder)

	else:
		checkpoint_path = os.path.join(CHECKPOINT_PATH, args.net, TIME_NOW)

	

	#create checkpoint folder to save model
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)

	checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

	if args.resume:
		best_weights = best_acc_weights(os.path.join(CHECKPOINT_PATH, args.net, recent_folder))
		if best_weights:
			weights_path = os.path.join(CHECKPOINT_PATH, args.net, recent_folder, best_weights)
			print('found best acc weights file:{}'.format(weights_path))
			print('load best training file to test acc...')
			net.load_state_dict(torch.load(weights_path))
			best_acc = eval_training(tb=False)
			print('best acc is {:0.2f}'.format(best_acc))

		recent_weights_file = most_recent_weights(os.path.join(CHECKPOINT_PATH, args.net, recent_folder))
		if not recent_weights_file:
			raise Exception('no recent weights file were found')
		weights_path = os.path.join(CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
		print('loading weights file {} to resume training.....'.format(weights_path))
		net.load_state_dict(torch.load(weights_path))

		resume_epoch = last_epoch(os.path.join(CHECKPOINT_PATH, args.net, recent_folder))

	writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, args.net, TIME_NOW))

	input_tensor = torch.Tensor(1, 3, args.resize, args.resize)

	# here index represents the network prediction and class represents original name of folder
	index_to_class = {index: classs for classs, index in datasets['training'].class_to_idx.items()}  

	class_to_label = load_object("class_to_label_22classes.pkl")

	class_names = [class_to_label[index_to_class[i]] for i in range(len(datasets['training'].classes))]

	if args.gpu:
		input_tensor = input_tensor.cuda()

	writer.add_graph(net, input_tensor)

	best_acc = 0.0
	
	save_hparams(args, writer)

	for epoch in range(1, args.epochs + 1):
		
		if epoch > args.warm:
			lr_scheduler.step(epoch)

		if args.resume:
			if epoch <= resume_epoch:
				continue

		train(epoch)
		acc = eval_training(epoch)

		#start to save best performance model after learning rate decay to 0.01
		if best_acc < acc:
			weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
			print('saving weights file to {}'.format(weights_path))
			torch.save(net.state_dict(), weights_path)
			best_acc = acc

	writer.close()
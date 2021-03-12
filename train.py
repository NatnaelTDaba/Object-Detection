""" Train PyTorch model.

Author: Natnael Daba

"""
import sys
import argparse
import time

import torch
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
		trained_samples=batch_index * args.b + len(images),
		total_samples=len(xView_train_loader.dataset)))

		#update training loss for each iteration
		writer.add_scalar('Train/loss', loss.item(), n_iter)

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

		test_loss += loss.item()
		_, preds = outputs.max(1)
		correct += preds.eq(labels).sum()

		all_predictions.extend(preds.cpu().tolist())
		all_targets.extend(labels.cpu().tolist())

	finish = time.time()

	matrix = confusion_matrix(all_targets, all_predictions)
	fig = plt.figure(figsize=[15,15])
	plt.imshow(matrix)
	plt.show()
	writer.add_figure('Test/Confusion Matrix', fig, epoch)
	if args.gpu:
		print('GPU INFO.....')
		print(torch.cuda.memory_summary(), end='')
	print('Evaluating Network.....')
	print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
		epoch,
		test_loss / len(xView_test_loader.dataset),
		correct.float() / len(xView_test_loader.dataset),
		finish - start
	))
	print()

	#add information to tensorboard
	if tb:
		writer.add_scalar('Test/Average loss', test_loss / len(xView_test_loader.dataset), epoch)
		writer.add_scalar('Test/Accuracy', correct.float() / len(xView_test_loader.dataset), epoch)

	return correct.float() / len(xView_test_loader.dataset)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-net', type=str, required=True, help='net type')
	parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
	parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
	parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
	args = parser.parse_args()

	print(args.net, args.gpu, args.b, args.lr)

	loader_arguments = {'training_transform':transforms.Compose([transforms.Resize((NEW_SIZE,NEW_SIZE)),
															transforms.ToTensor()]),
						'validation_transform':transforms.Compose([transforms.Resize((NEW_SIZE,NEW_SIZE)),
															transforms.ToTensor()]),
						'batch_size':args.b
						}

	loader = get_loader(**loader_arguments)
	
	xView_train_loader = loader['training']
	xView_test_loader = loader['validation']
	#first_batch = next(iter(xView_train_loader)) # For sanity check by overfitting a mini batch


	net = get_network(args)
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

	#use tensorboard
	
	if not os.path.exists(LOG_DIR):
		os.mkdir(LOG_DIR)

	writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, args.net, TIME_NOW))

	input_tensor = torch.Tensor(1, 3, NEW_SIZE, NEW_SIZE)

	if args.gpu:
		input_tensor = input_tensor.cuda()

	writer.add_graph(net, input_tensor)

	for epoch in range(1, EPOCH + 1):

		train(epoch)
		acc = eval_training(epoch)

	writer.close()
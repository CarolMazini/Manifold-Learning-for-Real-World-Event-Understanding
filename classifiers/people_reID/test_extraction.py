# -*- coding: utf-8 -*-

#Adapted from:
#Beyond Part Models: Person Retrieval with Refined Part Pooling and A Strong Convolutional Baseline
#Authors: Yifan Suny and Liang Zhengz and Yi Yangz and Qi Tianx and Shengjin Wang

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os,sys
import scipy.io
import yaml
from PIL import Image, ImageDraw
sys.path.insert(0, '../classifiers/')
from people_reID..model.model_test import PCB,PCB_test,ft_net



class MyDataset(Dataset):
	def __init__(self,base,dataframe, x_col, y_col,transform=None):
		self.input_images = dataframe[x_col]
		if y_col!=None:
			self.target_images = dataframe[y_col]
		else: 
			self.target_images = []
		self.transform = transform
		self.base = base

	def __getitem__(self, idx):
		if self.base != None:
			image = Image.open(self.base+self.input_images[idx])
		else:
			image = Image.open(self.input_images[idx])

		if len(self.target_images)>0:
			label  = self.target_images[idx]
		else:
			label = 0
		print(len(image.getbands()))
		#if len(image.getbands()) == 1:
			#print(image.shape)
		image = image.convert("RGB") 
			#print(image.shape)
		
		if self.transform:
			image = self.transform(image)
		
		return image,label

	def __len__(self):
		return len(self.input_images)



def extract_reID_features(config_path,path_weights, classe, dataset,dataframe,x_col,y_col, dir_out_features, gpu_ids):
	
	gallery = classe


	# #fp16
	# try:
	#	 from apex.fp16_utils import *
	# except ImportError: # will be 3.x series
	#	 print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
	# ######################################################################
	# Options
	# --------
	batchsize = 1
	###load config###
	# load the training config
	print(config_path)
	with open(config_path, 'r') as stream:
		config = yaml.load(stream)
	fp16 = False
	opt_PCB = True
	use_dense = False
	multi = False
	#opt.use_NAS = config['use_NAS']
	stride = config['stride']

	nclasses = 751

	str_ids = gpu_ids.split(',')


	gpu_ids = []
	for str_id in str_ids:
		id = int(str_id)
		if id >=0:
			gpu_ids.append(id)

	# set gpu ids
	if len(gpu_ids)>0:
		torch.cuda.set_device(gpu_ids[0])
		cudnn.benchmark = True

	######################################################################
	# Load Data
	# ---------
	#
	# We will use torchvision and torch.utils.data packages for loading the
	# data.
	#
	data_transforms = transforms.Compose([
			transforms.Resize((256,128), interpolation=3),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	############### Ten Crop
			#transforms.TenCrop(224),
			#transforms.Lambda(lambda crops: torch.stack(
			#   [transforms.ToTensor()(crop)
			#	  for crop in crops]
			# )),
			#transforms.Lambda(lambda crops: torch.stack(
			#   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
			#	   for crop in crops]
			# ))
	])

	if opt_PCB:
		data_transforms = transforms.Compose([
			transforms.Resize((384,192), interpolation=3),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])



	if multi:
		image_datasets = MyDataset(None,dataframe,x_col,y_col, transform=data_transforms)
		#image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in [gallery]}
		dataloaders = torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,shuffle=False, num_workers=16)
	else:
		image_datasets = MyDataset(None,dataframe,x_col,y_col, transform=data_transforms)
		dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize,shuffle=False, num_workers=16)
	class_names = image_datasets.target_images
	use_gpu = torch.cuda.is_available()

	######################################################################
	# Load model
	#---------------------------
	def load_network(network):
		save_path = path_weights
		print(save_path)
		network.load_state_dict(torch.load(save_path))
		return network


	######################################################################
	# Extract feature
	# ----------------------
	#
	# Extract feature from  a trained model.
	#
	def fliplr(img):
		'''flip horizontal'''
		inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
		img_flip = img.index_select(3,inv_idx)
		return img_flip

	def extract_feature(model,dataloaders):
		features = torch.FloatTensor()
		count = 0
		for data in dataloaders:

			img, label = data
			n, c, h, w = img.size()
			count += n
			print(count)
			ff = torch.FloatTensor(n,512).zero_()

			if opt_PCB:
				ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
			for i in range(2):
				if(i==1):
					img = fliplr(img)
				input_img = Variable(img.cuda())
			
				outputs = model(input_img)
				f = outputs.data.cpu().float()
				ff = ff+f
			# norm feature
			if opt_PCB:
				# feature size (n,2048,6)
				# 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
				# 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
				fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
				ff = ff.div(fnorm.expand_as(ff))
				ff = ff.view(ff.size(0), -1)
			else:
				fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
				ff = ff.div(fnorm.expand_as(ff))

			features = torch.cat((features,ff), 0)

		return features



	def get_id(img_path):
		camera_id = []
		labels = []
		for path, v in img_path:
			filename_folder = path.split('/')[-2]
			filename = os.path.basename(path)
			if filename_folder == 'peopleRelevant':
				labels.append(0)
			elif filename_folder == 'notRelevant':
				labels.append(1)
			else: #query
				labels.append(2)

		return labels

	gallery_path = image_datasets.input_images


	######################################################################
	# Load Collected data Trained model

	print('-------test-----------')
	# if use_dense:
	#	 model_structure = ft_net_dense(nclasses)
	# else:
	#	 model_structure = ft_net(nclasses, stride = stride)



	if opt_PCB:
		model_structure = PCB(nclasses)

	model = load_network(model_structure)

	# Remove the final fc layer and classifier layer
	if opt_PCB:
		model = PCB_test(model)
	else:

		model.classifier.classifier = nn.Sequential()

	# Change to test mode
	model = model.eval()
	if use_gpu:
		model = model.cuda()

	# Extract feature
	with torch.no_grad():
		gallery_feature = extract_feature(model,dataloaders)

	not_relevant = []
	relevant = []

	filenames_not = []
	filenames_relevant = []

	allFeatures = []
	allFilenames = []

	for i in range(0,len(gallery_path)):
		allFeatures.append(np.array(gallery_feature[i]))
		allFilenames.append(np.array(gallery_path[i]))

	print(np.array(allFeatures).shape)
	print(gallery)

	#np.save(dir_out_features+'all_'+gallery+'_people.npy', np.array(allFeatures))
	#np.save(dir_out_features+'all_'+gallery+'_people_filenames.npy', np.array(allFilenames))
	return np.array(allFeatures),np.array(allFilenames)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import random
import os,sys
import shutil

##################################MAIN#####################################

def save_train_val(dir,in_csv):


	name_path_save_val = os.path.join(dir,'val/')
	if not os.path.isdir(name_path_save_val):
		os.mkdir(name_path_save_val)
	name_path_save_train = os.path.join(dir,'train/')
	if not os.path.isdir(name_path_save_train):
		os.mkdir(name_path_save_train)

	## val
	df = pd.read_csv(os.path.join(in_csv,'val_.csv'))

	print(df.label.values)

	for i in range(len(df.label.values)):
		if not os.path.isdir(name_path_save_val+str(df.label.values[i])):
			os.mkdir(name_path_save_val+str(df.label.values[i]))
		name_image = df.path.values[i].split('/')[-2]+'-'+df.path.values[i].split('/')[-1]
		shutil.copyfile(df.path.values[i], name_path_save_val+str(df.label.values[i])+'/'+name_image)
		print(name_image)

	## train
	df = pd.read_csv(os.path.join(in_csv,'train_.csv'))

	print(df.label.values)

	for i in range(len(df.label.values)):
		if not os.path.isdir(name_path_save_train+str(df.label.values[i])):
			os.mkdir(name_path_save_train+str(df.label.values[i]))
		name_image = df.path.values[i].split('/')[-2]+'-'+df.path.values[i].split('/')[-1]
		shutil.copyfile(df.path.values[i], name_path_save_train+str(df.label.values[i])+'/'+name_image)
		print(name_image)

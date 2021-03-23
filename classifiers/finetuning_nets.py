import numpy as np
import math
import shutil
import os
import glob
import random
import os, sys, stat
from os import *
import random
import pandas as pd 
sys.path.insert(0, '../classifiers/')
from people_reID.train_finetune import finetune_reID_network
import cnn_finetune
import argparse
	

def finetune_models(name,df_train,df_val,path_base,name_objects, name_places,weights_ini_imagenet,weights_ini_places):

	epochs = 50

	# objects imagenet
	model_obj = cnn_finetune.loadNetwork('objects',weights_ini_imagenet)
	print(model_obj.summary())
	data_train = cnn_finetune.generateData_fromFile(299,df_train,'objects')
	data_val = cnn_finetune.generateData_fromFile(299,df_val,'objects')
	model_obj = cnn_finetune.train_model(model_obj,data_train,data_val,epochs,os.path.join(path_base,'best_'+name_objects))
	model_obj.save_weights(os.path.join(path_base,name_objects))

	# places imagenet
	model_pla = cnn_finetune.loadNetwork('places',weights_ini_places)
	print(model_pla.summary())
	data_train = cnn_finetune.generateData(224,df_train,'places')
	data_val = cnn_finetune.generateData(224,df_val,'places')
	model_pla = cnn_finetune.train_model(model_pla,data_train,data_val,epochs,os.path.join(path_base,'best_'+name_places))
	model_pla.save_weights(os.path.join(path_base,name_places))

def modify_dataframe(df, col,old, new):
	df[col] = df[col].str.replace(old,new)
	return df




def finetune_all_nets(path_datasets, weights_ini_imagenet,weights_ini_places,weights_ini_PCB,aug,gpu_ids):


	filename = '../dataset'
	
	name_final_folder = dataset
	classe = ['positive','negative']
	finetuned_weights = '../out_files/checkpoints/finetuned'
	if not os.path.isdir(finetuned_weights):
		os.mkdir(finetuned_weights)
	

	finetuned_weights = os.path.join(finetuned_weights, name_final_folder)
	if not os.path.isdir(finetuned_weights):
		os.mkdir(finetuned_weights)

	
	

	df_train = pd.read_csv(os.path.join(filename, dataset,'train'+aug+'.csv'), header=0, dtype = str)
	df_val = pd.read_csv(os.path.join(filename, dataset,'val'+aug+'.csv'), header=0, dtype = str)	
	
	#------------------	
	#----------------
	#train finetuned models
	
	finetune_models(os.path.join(filename, dataset),df_train, df_val,finetuned_weights,'weights_imagenet.h5', 'weights_places.h5',weights_ini_imagenet,weights_ini_places)

	finetune_reID_network(weights_ini_PCB,df_train, df_val,'path','label', finetuned_weights,gpu_ids)


##################################MAIN#####################################

if __name__ == '__main__':

	

	parser = argparse.ArgumentParser(description='Networks fine-tuning.')
	parser.add_argument('--dataset', dest='dataset', type=str,help='include the name of the dataset to extract image features',default='bombing')
	parser.add_argument('--aug', dest='aug', type=str,help='include _aug to use augmented data', default='')
	parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	dataset = args.dataset #'wedding', 'fire', 'bombing', 'museu_nacional' or 'bangladesh_fire'
	aug=args.aug

	set_gpu(gpu_ids)
        finetune_all_nets('../dataset/','../extract_features/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5', '../extract_features/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5','./people_reID/model/net_last.pth',aug,gpu_ids)	

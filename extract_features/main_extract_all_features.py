import numpy as np
import math
import shutil
import os
import glob
import random
import os, sys, stat
from os import *
import extract_objects_places_features
import random
import pandas as pd
from save_inDir import * 
import argparse
from shutil import copyfile
sys.path.insert(0, '../classifiers/')
from people_reID.test_extraction import extract_reID_features




#extracts and saves the features from posterior analysis
def extracting_features_set(aug,dataset, gpu_ids, finetuned=False):


	filename = '../dataset/'
	
	name_final_folder = dataset+aug
	classe = ['positive','negative']
	finetuned_weights = '../out_files/checkpoints/finetuned'
	if not os.path.isdir(finetuned_weights):
		os.mkdir(finetuned_weights)


	dir_features = '../out_files/features'
	pd.set_option('display.max_colwidth', -1)

	

	finetuned_weights = os.path.join(finetuned_weights, name_final_folder)
	if not os.path.isdir(finetuned_weights):
		os.mkdir(finetuned_weights)



	feature_extractor_places = extract_objects_places_features.loadNetworkPlaces(1,finetuned_weights+'best_weights_places.h5',finetuned)
	feature_extractor_imagenet = extract_objects_places_features.loadNetworkImagenet(1,finetuned_weights+'best_weights_imagenet.h5',finetuned)

	
	#------------------	
	#----------------
	#extract features objects,  places and people -- train/val/test
	
	for n in ['train','val','test']:

		out_features = os.path.join(dir_features, name_final_folder)
		if not os.path.isdir(out_features):
			os.mkdir(out_features)
		if n == 'test': # test do not have augmentation
			name = os.path.join(filename, dataset,n)
		else:
			name = os.path.join(filename, dataset,n+aug)
		
		
		df_paths = pd.read_csv(name+'.csv', header=0) 	
		 
		labels = np.array(df_paths.label)
		negative_labels = np.ones(len(labels)) - labels
		
		labels = list(map(bool,labels))
		negative_labels = list(map(bool,negative_labels))
	 
		#----------------
		#extract features objects and places

		
		generatorPlaces= extract_objects_places_features.generateData_fromFile(224,df_paths)
		names = generatorPlaces.filenames
		generatorImagenet = extract_objects_places_features.generateData_fromFile(299,df_paths)
		features_imagenet, features_places = extract_objects_places_features.getFeatures(generatorImagenet,generatorPlaces,feature_extractor_imagenet,feature_extractor_places)
		

		np.save(out_features+'/positive_'+n+'_names'+aug+'.npy', np.array(names)[labels])
		np.save(out_features+'/negative_'+n+'_names'+aug+'.npy', np.array(names)[negative_labels])
		np.save(out_features+'/positive_'+n+'_imagenet'+aug+'.npy', np.array(features_imagenet)[labels])
		np.save(out_features+'/negative_'+n+'_imagenet'+aug+'.npy', np.array(features_imagenet)[negative_labels])
		np.save(out_features+'/positive_'+n+'_places'+aug+'.npy', np.array(features_places)[labels])
		np.save(out_features+'/negative_'+n+'_places'+aug+'.npy', np.array(features_places)[negative_labels])
		

		#----------------
		#extract features from PCB
		features_people, names = extract_reID_features('../classifiers/people_reID/opts.yaml','../classifiers/people_reID/model/net_last.pth', n, dataset,df_paths,'path','label', out_features, gpu_ids)
		
		np.save(out_features+'/positive_'+n+'_names_people'+aug+'.npy', np.array(names)[labels])
		np.save(out_features+'/negative_'+n+'_names_people'+aug+'.npy', np.array(names)[negative_labels])
		np.save(out_features+'/positive_'+n+'_people'+aug+'.npy', np.array(features_people)[labels])
		np.save(out_features+'/negative_'+n+'_people'+aug+'.npy', np.array(features_people)[negative_labels])
			
	
	#------------------	 
	#----------------
	#extract features objects,  places and people -- unlabeled test images (museum and bangladesh)

	if dataset == 'museu_nacional' or dataset == 'bangladesh_fire':
		out_features = os.path.join(dir_features, name_final_folder)
		name = os.path.join(filename, dataset,'test_real')
		df_paths = pd.read_csv(name+'.csv', header=0)  
		
		generatorPlaces= extract_objects_places_features.generateData_fromFile(224,df_paths)
		names = generatorPlaces.filenames
		generatorImagenet = extract_objects_places_features.generateData_fromFile(299,df_paths)
		features_imagenet, features_places = extract_objects_places_features.getFeatures(generatorImagenet,generatorPlaces,feature_extractor_imagenet,feature_extractor_places)
	
		np.save(out_features+'/test_real_names'+aug+'.npy', np.array(names))
		np.save(out_features+'/test_real_imagenet'+aug+'.npy', np.array(features_imagenet))
		np.save(out_features+'/test_real_places'+aug+'.npy', np.array(features_places))
	
		features_people, names = extract_reID_features('../classifiers/people_reID/opts.yaml','../classifiers/people_reID/model/net_last.pth', 'test_real', dataset,df_paths,'path',None, out_features, gpu_ids)
	
		np.save(out_features+'_test_real_names_people'+aug+'.npy', np.array(names))
		np.save(out_features+'_test_real_people'+aug+'.npy', np.array(features_people))	

	


##################################MAIN#####################################

if __name__ == '__main__':

	

	parser = argparse.ArgumentParser(description='Features Extraction.')
	parser.add_argument('--dataset', dest='dataset', type=str,help='include the name of the dataset to extract image features',default='bombing')
	parser.add_argument('--finetuned', action='store_true',help='determine if finetuned weights will be used')
	parser.add_argument('--aug', dest='aug', type=str,help='include _aug to use augmented data', default='')
	parser.add_argument('--gpu_ids',default='0', type=str,help='number of gpu to be used')
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	dataset = args.dataset #'wedding', 'fire', 'bombing', 'museu_nacional' or 'bangladesh_fire'
	finetuned = args.finetuned # if true, load the finetuned weights to the network
	print(finetuned)
	aug=args.aug

	extract_objects_places_features.set_gpu(gpu_ids)
	extracting_features_set(aug,dataset, gpu_ids, finetuned)


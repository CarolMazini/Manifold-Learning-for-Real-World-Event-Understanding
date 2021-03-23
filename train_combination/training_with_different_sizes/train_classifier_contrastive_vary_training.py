#!/usr/bin/env python3
import numpy as np
import os,sys
import glob
sys.path.insert(0, '../../classifiers/')
from create_network_siamese import *
import random
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import argparse


def choice_training(names_noAug, names, num_train_noAug):

	final_names = []
	final_features = []

	#sampling train and val sets
	names_chosen = names_noAug[num_train_noAug]
	
	for name_chosen in names_chosen:
		n1 = (name_chosen.split('/')[-1]).split('.')[-2]
		n2 = name_chosen.split('/')[-2]
		for index in range(len(names)):
			if (n1 in names[index]) and (n2 in names[index]):
				final_names.append(names[index])
				final_features.append(index)
	print(len(final_names))

	return np.array(final_names), np.array(final_features)

##################################MAIN#####################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Training with small training sets the combination with contrastive loss. The code uses augmented data')
	parser.add_argument('--dataset', dest='dataset', type=str,help='include the name of the dataset to extract image features',default='bombing')
	parser.add_argument('--arch', dest='arch', type=int,help='type of the architecture. Possible values: 0,1,2 or 3 for 512_128, 512_128_64, 1024_512 or 1024_512_128, respectively',default='2')	
	#parser.add_argument('--aug', dest='aug', type=str,help='include _aug to use augmented data', default='')
	parser.add_argument('--gpu_ids',default='0', type=str,help='number of gpu to be used')
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	dataset = args.dataset #'wedding', 'fire', 'bombing', 'museu_nacional' or 'bangladesh_fire'
	aug='_aug'
	t = args.arch

	if tf.get_default_session() is not None:
		status = tf.get_default_session().TF_NewStatus()
		tf.get_default_session().TF_DeleteDeprecatedSession(self._session, status)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = False #avoid getting all available memory in GPU
	config.gpu_options.per_process_gpu_memory_fraction = 0.2  #uncomment to limit GPU allocation
	config.gpu_options.visible_device_list = gpu_ids  #set which GPU to use
	set_session(tf.Session(config=config))

	names_arch = ['512_128', '512_128_64', '1024_512', '1024_512_128']
	#type_architecture = [0,1,2,3]

	filename = '../../dataset'
	dir_features = '../../out_files/features'
	dir_checkpoints = '../../out_files/checkpoints'

	train_numbers = [10,20,50,100,200]
	name_architecture = names_arch[t]
	type_architecture = t

	dir_out_features = os.path.join(dir_features, 'contrastive')
	dir_out_model = os.path.join(dir_checkpoints, 'contrastive')
	if not os.path.isdir(dir_out_features):
		os.mkdir(dir_out_features)
	if not os.path.isdir(dir_out_model):
		os.mkdir(dir_out_model)


	#geral_datasets = ['wedding','fire','bombing']
	#geral_aug = ['_aug'] 

	name_final_folder = dataset
	name_final_folder = name_final_folder+aug
	in_features = os.path.join(dir_features, name_final_folder)
	base_features = os.path.join(dir_features, dataset)


	#----------------	
	#----------------
	#reading features
	positive_places_train = np.load(in_features+'/positive_train_places.npy')
	positive_places_val = np.load(in_features+'/positive_val_places.npy')
	positive_places_test = np.load(in_features+'/positive_test_places.npy')
	positive_imagenet_train = np.load(in_features+'/positive_train_imagenet.npy')
	positive_imagenet_val = np.load(in_features+'/positive_val_imagenet.npy')
	positive_imagenet_test = np.load(in_features+'/positive_test_imagenet.npy')
	positive_reID_train = np.load(in_features+'/positive_train_people.npy')
	positive_reID_val = np.load(in_features+'/positive_val_people.npy')
	positive_reID_test = np.load(in_features+'/positive_test_people.npy')

	negative_places_train = np.load(in_features+'/negative_train_places.npy')
	negative_places_val = np.load(in_features+'/negative_val_places.npy')
	negative_places_test = np.load(in_features+'/negative_test_places.npy')
	negative_imagenet_train = np.load(in_features+'/negative_train_imagenet.npy')
	negative_imagenet_val = np.load(in_features+'/negative_val_imagenet.npy')
	negative_imagenet_test = np.load(in_features+'/negative_test_imagenet.npy')
	negative_reID_train = np.load(in_features+'/negative_train_people.npy')
	negative_reID_val = np.load(in_features+'/negative_val_people.npy')
	negative_reID_test = np.load(in_features+'/negative_test_people.npy')

	print(len(negative_imagenet_val))

	

	train_names_noAug = np.load(base_features+'/positive_train_names.npy')
	train_names_aug = np.load(in_features+'/positive_train_names.npy')

	val_names_noAug = np.load(base_features+'/positive_val_names.npy')
	val_names_aug = np.load(in_features+'/positive_val_names.npy')


	iterations = 10
	train_samples = []
	val_samples = []
	num_val = int(train_numbers[-1]*0.2)
	if num_val>len(val_names_noAug):
		num_val =len(val_names_noAug)
	print('Num val:', num_val)
	for i in range(iterations):
		random.seed(a=i)
		train_samples.append(random.sample(range(0,len(train_names_noAug)), train_numbers[-1]))
		val_samples.append(random.sample(range(0,len(val_names_noAug)), num_val))

	#----------------	
	#----------------
	#output model directory
	out_path_model = os.path.join(dir_out_model, name_final_folder)
	if not os.path.isdir(out_path_model):
		os.mkdir(out_path_model)


	out_path_features = os.path.join(dir_out_features, name_final_folder)
	if not os.path.isdir(out_path_features):
		os.mkdir(out_path_features)

	np.save(os.path.join(out_path_features,'train_indexes_'+dataset+aug+'.npy'),train_samples)
	np.save(os.path.join(out_path_features,'val_indexes_'+dataset+aug+'.npy'),val_samples)


	for i in range(iterations):
		print('------------------')
		print('------------------')
		print('Iterations:', i)

		for num_train in train_numbers:

			if num_train < len(train_names_noAug):
				print('------------------')
				print('------------------')
				print('Num train:', num_train)
				num_val = int(num_train*0.2)
				if num_val>len(val_names_noAug):
					num_val =len(val_names_noAug)
				print('Num val:', num_val)

				if not os.path.isfile(os.path.join(out_path_model,str(i)+'_modelo_final_'+name_architecture+'_split_random10timesEach_'+str(num_train)+'_'+str(num_val)+aug+'.h5')):
	
		
					relevant_train = len(positive_places_train)
					relevant_val = len(positive_places_val)
					nonrelevant_train = len(negative_places_train)
					nonrelevant_val = len(negative_places_val)

					print('Relevant train: ', relevant_train)
					print('NonRelevant train: ', nonrelevant_train)
					print('Relevant val: ', relevant_val)
					print('NonRelevant val: ', nonrelevant_val)
					print('Relevant test: ', len(positive_places_test))
					print('NonRelevant test: ', len(negative_places_test))

					#params
					epochs = 50

					#----------------	
					#----------------
					#sampling train and val sets
					train_names, index_positive_train = choice_training(train_names_noAug, train_names_aug, train_samples[i][0:num_train])
					val_names, index_positive_val = choice_training(val_names_noAug, val_names_aug, val_samples[i][0:num_val])
					
					random.seed(a=i)
					
					index_negative_train = random.sample(range(0,nonrelevant_train), np.min([len(index_positive_train),len(negative_imagenet_train)]))
					index_negative_val = random.sample(range(0,nonrelevant_val), np.min([len(index_positive_val),len(negative_imagenet_val)]))


					train_positive_data = [positive_places_train[index_positive_train],positive_imagenet_train[index_positive_train],positive_reID_train[index_positive_train]]
					train_negative_data = [negative_places_train[index_negative_train],negative_imagenet_train[index_negative_train],negative_reID_train[index_negative_train]]

					val_positive_data = [positive_places_val[index_positive_val],positive_imagenet_val[index_positive_val],positive_reID_val[index_positive_val]]
					val_negative_data = [negative_places_val[index_negative_val],negative_imagenet_val[index_negative_val],negative_reID_val[index_negative_val]]

					#----------------	
					#----------------
					#training and saving models
					model2 = train_siamese(train_positive_data, train_negative_data, val_positive_data, val_negative_data, [len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])], os.path.join(out_path_model,str(i)+'_modelo_best_'+name_architecture+'_split_random10timesEach_'+str(num_train)+'_'+str(num_val)+aug+'.h5'), type_architecture)


					model2.save_weights(os.path.join(out_path_model,str(i)+'_modelo_final_'+name_architecture+'_split_random10timesEach_'+str(num_train)+'_'+str(num_val)+aug+'.h5'))


					path_model = str(i)+'_modelo_best_'+name_architecture+'_split_random10timesEach_'+str(num_train)+'_'+str(num_val)+aug

				

					train_positive_data = [positive_places_train,positive_imagenet_train,positive_reID_train]
					train_negative_data = [negative_places_train,negative_imagenet_train,negative_reID_train]

					val_positive_data = [positive_places_val,positive_imagenet_val,positive_reID_val]
					val_negative_data = [negative_places_val,negative_imagenet_val,negative_reID_val]

					test_positive_data = [positive_places_test,positive_imagenet_test,positive_reID_test]
					test_negative_data = [negative_places_test,negative_imagenet_test,negative_reID_test]


					#----------------	
					#----------------
					#extracting features
					features_positive_train, features_negative_train = extract_features_siamese(train_positive_data, train_negative_data, 'layer_name',[len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])],os.path.join(out_path_model,path_model+'.h5'), type_architecture)
					features_positive_val, features_negative_val = extract_features_siamese(val_positive_data, val_negative_data, 'layer_name',[len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])],os.path.join(out_path_model,path_model+'.h5'), type_architecture)
					features_positive_test, features_negative_test = extract_features_siamese(test_positive_data, test_negative_data, 'layer_name',[len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])],os.path.join(out_path_model,path_model+'.h5'), type_architecture)


					#----------------	
					#----------------
					#saving extracted features
					np.save(os.path.join(out_path_features,str(i)+'_negative_'+str(num_train)+'_'+str(num_val)+'_'+dataset+aug+'_train_'+name_architecture+'.npy'),features_negative_train)
					np.save(os.path.join(out_path_features,str(i)+'_negative_'+str(num_train)+'_'+str(num_val)+'_'+dataset+aug+'_val_'+name_architecture+'.npy'),features_negative_val)
					np.save(os.path.join(out_path_features,str(i)+'_negative_'+str(num_train)+'_'+str(num_val)+'_'+dataset+aug+'_test_'+name_architecture+'.npy'),features_negative_test)
					np.save(os.path.join(out_path_features,str(i)+'_positive_'+str(num_train)+'_'+str(num_val)+'_'+dataset+aug+'_train_'+name_architecture+'.npy'),features_positive_train)
					np.save(os.path.join(out_path_features,str(i)+'_positive_'+str(num_train)+'_'+str(num_val)+'_'+dataset+aug+'_val_'+name_architecture+'.npy'),features_positive_val)
					np.save(os.path.join(out_path_features,str(i)+'_positive_'+str(num_train)+'_'+str(num_val)+'_'+dataset+aug+'_test_'+name_architecture+'.npy'),features_positive_test)
				
				else:
					print('Already extracted:', i)



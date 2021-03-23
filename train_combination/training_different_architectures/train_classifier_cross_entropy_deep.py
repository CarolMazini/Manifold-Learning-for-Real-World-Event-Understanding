#!/usr/bin/env python3
import numpy as np
import os,sys
import glob
sys.path.insert(0, '../../classifiers/')
from create_network import *
import random
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import argparse

##################################MAIN#####################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Training combination with cross-entropy loss.')
	parser.add_argument('--dataset', dest='dataset', type=str,help='include the name of the dataset to extract image features',default='bombing')
	parser.add_argument('--arch', dest='arch', type=int,help='type of the architecture. Possible values: 0,1,2 or 3 for 512_128, 512_128_64, 1024_512 or 1024_512_128, respectively',default='2')
	parser.add_argument('--aug', dest='aug', type=str,help='include _aug to use augmented data', default='')
	parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: number of gpu to be used')
	args = parser.parse_args()
	

	gpu_ids = args.gpu_ids
	dataset = args.dataset #'wedding', 'fire', 'bombing', 'museu_nacional' or 'bangladesh_fire'
	aug=args.aug
	t = args.arch
	
	if tf.get_default_session() is not None:
		status = tf.get_default_session().TF_NewStatus()
		tf.get_default_session().TF_DeleteDeprecatedSession(self._session, status)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = False	#avoid getting all available memory in GPU
	config.gpu_options.per_process_gpu_memory_fraction = 0.2  #uncomment to limit GPU allocation
	config.gpu_options.visible_device_list = gpu_ids  #set which GPU to use
	set_session(tf.Session(config=config))
	

	names_arch = ['512_128', '512_128_64', '1024_512', '1024_512_128']
	#type_architecture = [0,1,2,3]

	filename = '../../dataset'
	dir_features = '../../out_files/features'
	dir_checkpoints = '../../out_files/checkpoints'
	dir_out_features = os.path.join(dir_features, 'cross_entropy')
	dir_out_model = os.path.join(dir_checkpoints, 'cross_entropy')
	
	if not os.path.isdir(dir_out_features):
		os.mkdir(dir_out_features)
	if not os.path.isdir(dir_out_model):
		os.mkdir(dir_out_model)


	#geral_datasets = ['wedding','fire','bombing', 'museu_nacional','bangladesh_fire']
	#geral_aug = ['','_aug'] 

	
	name_final_folder = dataset
	name_final_folder = name_final_folder+aug
	in_features = os.path.join(dir_features, name_final_folder)


	#----------------
	#----------------
	#reading features
	positive_places_train = np.load(in_features+'/positive_train_places'+aug+'.npy')
	positive_places_val = np.load(in_features+'/positive_val_places'+aug+'.npy')
	positive_places_test = np.load(in_features+'/positive_test_places'+aug+'.npy')
	positive_imagenet_train = np.load(in_features+'/positive_train_imagenet'+aug+'.npy')
	positive_imagenet_val = np.load(in_features+'/positive_val_imagenet'+aug+'.npy')
	positive_imagenet_test = np.load(in_features+'/positive_test_imagenet'+aug+'.npy')
	positive_reID_train = np.load(in_features+'/positive_train_people'+aug+'.npy')
	positive_reID_val = np.load(in_features+'/positive_val_people'+aug+'.npy')
	positive_reID_test = np.load(in_features+'/positive_test_people'+aug+'.npy')

	negative_places_train = np.load(in_features+'/negative_train_places'+aug+'.npy')
	negative_places_val = np.load(in_features+'/negative_val_places'+aug+'.npy')
	negative_places_test = np.load(in_features+'/negative_test_places'+aug+'.npy')
	negative_imagenet_train = np.load(in_features+'/negative_train_imagenet'+aug+'.npy')
	negative_imagenet_val = np.load(in_features+'/negative_val_imagenet'+aug+'.npy')
	negative_imagenet_test = np.load(in_features+'/negative_test_imagenet'+aug+'.npy')
	negative_reID_train = np.load(in_features+'/negative_train_people'+aug+'.npy')
	negative_reID_val = np.load(in_features+'/negative_val_people'+aug+'.npy')
	negative_reID_test = np.load(in_features+'/negative_test_people'+aug+'.npy')
	
	if dataset== 'museu_nacional' or dataset =='bangladesh_fire':
	    places_test_real = np.load(in_features+'/test_real_places'+aug+'.npy')
	    imagenet_test_real = np.load(in_features+'/test_real_imagenet'+aug+'.npy')
	    reID_test_real = np.load(in_features+'/test_real_people'+aug+'.npy')
	

	print(len(positive_places_train))
	print('------------------')
	print('------------------')
	print('------------------')
	print('Dataset:', dataset)
	print('Aug:', aug)
	print('------------------')
	print('------------------')

	
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
	tam_train = np.min([relevant_train,nonrelevant_train])
	tam_val = np.min([relevant_val,nonrelevant_val]) #20% of training
	

	random.seed(a=0)

	#----------------	
	#----------------
	#sampling train and val sets
	index_positive_train = random.sample(range(0,relevant_train), tam_train)
	index_negative_train = random.sample(range(0,nonrelevant_train), tam_train)
	index_positive_val = random.sample(range(0,relevant_val), tam_val)
	index_negative_val = random.sample(range(0,nonrelevant_val), tam_val)
		

	train_positive_data = [positive_places_train[index_positive_train],positive_imagenet_train[index_positive_train],positive_reID_train[index_positive_train]]
	train_negative_data = [negative_places_train[index_negative_train],negative_imagenet_train[index_negative_train],negative_reID_train[index_negative_train]]

	val_positive_data = [positive_places_val[index_positive_val],positive_imagenet_val[index_positive_val],positive_reID_val[index_positive_val]]
	val_negative_data = [negative_places_val[index_negative_val],negative_imagenet_val[index_negative_val],negative_reID_val[index_negative_val]]


	name_architecture = names_arch[t]

	#----------------	
	#----------------
	#training and saving models
	out_path_model = os.path.join(dir_out_model, name_final_folder)
	if not os.path.isdir(out_path_model):
		os.mkdir(out_path_model)

	if t == 0:
		# network definition
		model = define_neural_net_functional(len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0]),2)
	elif t == 1:
		# network definition
		model = define_neural_net_functional_deep1(len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0]),2)
	elif t == 2:
		# network definition
		model = define_neural_net_functional_deep2(len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0]),2)
	else:
		# network definition
		model = define_neural_net_functional_deep3(len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0]),2)				

	model2 = train_model(model,train_positive_data,train_negative_data,val_positive_data,val_negative_data,epochs,out_path_model+'/modelo_best_'+name_architecture+'_split_'+str(tam_train)+'_'+str(tam_val)+aug+'.h5')


	model2.save_weights(out_path_model+'/modelo_'+name_architecture+'_split_'+str(tam_train)+'_'+str(tam_val)+'_'+str(epochs)+aug+'.h5')

	#extract features

	path_model = 'modelo_best_'+name_architecture+'_split_'+str(tam_train)+'_'+str(tam_val)+aug


	extraction_model = extract_features(model2,out_path_model+'/'+path_model+'.h5')

	people_train= np.concatenate([positive_reID_train, negative_reID_train], axis=0)
	imagenet_train= np.concatenate([positive_imagenet_train, negative_imagenet_train], axis=0)
	places_train= np.concatenate([positive_places_train, negative_places_train], axis=0)

	people_val= np.concatenate([positive_reID_val, negative_reID_val], axis=0)
	imagenet_val= np.concatenate([positive_imagenet_val, negative_imagenet_val], axis=0)
	places_val= np.concatenate([positive_places_val, negative_places_val], axis=0)


	people_test= np.concatenate([positive_reID_test, negative_reID_test], axis=0)
	imagenet_test= np.concatenate([positive_imagenet_test, negative_imagenet_test], axis=0)
	places_test= np.concatenate([positive_places_test, negative_places_test], axis=0)


	#----------------	
	#----------------
	#extracting features	
	features_positive_train = extraction_model.predict({'places': positive_places_train, 'objects': positive_imagenet_train, 'people': positive_reID_train},verbose=1)
	features_negative_train = extraction_model.predict({'places': negative_places_train, 'objects': negative_imagenet_train, 'people': negative_reID_train},verbose=1)

	features_positive_val = extraction_model.predict({'places': positive_places_val, 'objects': positive_imagenet_val, 'people': positive_reID_val},verbose=1)
	features_negative_val = extraction_model.predict({'places': negative_places_val, 'objects': negative_imagenet_val, 'people': negative_reID_val},verbose=1)

	features_positive_test = extraction_model.predict({'places': positive_places_test, 'objects': positive_imagenet_test, 'people': positive_reID_test},verbose=1)
	features_negative_test = extraction_model.predict({'places': negative_places_test, 'objects': negative_imagenet_test, 'people': negative_reID_test},verbose=1)

	
	if dataset== 'museu_nacional' or dataset =='bangladesh_fire':
	    features_test_real = extraction_model.predict({'places': places_test_real, 'objects': imagenet_test_real, 'people': reID_test_real},verbose=1)
	
	#----------------	
	#----------------
	#saving extracted features
	out_path_features = os.path.join(dir_out_features, name_final_folder)
	if not os.path.isdir(out_path_features):
		os.mkdir(out_path_features)

	np.save(os.path.join(out_path_features,name_architecture+'_negative_'+dataset+aug+'_train.npy'),features_negative_train)
	np.save(os.path.join(out_path_features,name_architecture+'_negative_'+dataset+aug+'_val.npy'),features_negative_val)
	np.save(os.path.join(out_path_features,name_architecture+'_negative_'+dataset+aug+'_test.npy'),features_negative_test)
	np.save(os.path.join(out_path_features,name_architecture+'_positive_'+dataset+aug+'_train.npy'),features_positive_train)
	np.save(os.path.join(out_path_features,name_architecture+'_positive_'+dataset+aug+'_val.npy'),features_positive_val)
	np.save(os.path.join(out_path_features,name_architecture+'_positive_'+dataset+aug+'_test.npy'),features_positive_test)

	
	if dataset== 'museu_nacional' or dataset =='bangladesh_fire':
	    np.save(os.path.join(out_path_features,name_architecture+dataset+aug+'_test_real.npy'),features_test_real)
	


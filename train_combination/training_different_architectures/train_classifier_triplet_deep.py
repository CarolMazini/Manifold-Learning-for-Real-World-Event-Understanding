#!/usr/bin/env python3
import numpy as np
import os,sys
import glob
sys.path.insert(0, '../../classifiers/')
from create_network_siamese_triplet import *
import random
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import argparse


##################################MAIN#####################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Training combination with triplet loss.')
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
	dir_out_features = os.path.join(dir_features, 'triplet')
	dir_out_model = os.path.join(dir_checkpoints, 'triplet')
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


	val_label = np.zeros(2*tam_val)
	val_data_Pl = []
	val_data_Ob = []
	val_data_Pe = []
	for i in range(tam_val):
		val_data_Pl.append(val_positive_data[0][i])
		val_data_Ob.append(val_positive_data[1][i])
		val_data_Pe.append(val_positive_data[2][i])
		val_label[2*i] = 1
		val_data_Pl.append(val_negative_data[0][i])
		val_data_Ob.append(val_negative_data[1][i])
		val_data_Pe.append(val_negative_data[2][i])
	val_data = [np.array(val_data_Pl),np.array(val_data_Ob),np.array(val_data_Pe)]


	name_architecture = names_arch[t]

	#----------------	
	#----------------
	#training and saving models
	out_path_model = os.path.join(dir_out_model, name_final_folder)
	if not os.path.isdir(out_path_model):
		os.mkdir(out_path_model)

	train_siamese_triplet(train_positive_data, train_negative_data, val_data, val_label, [len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])],64,out_path_model,'modelo_'+name_architecture+'_split_triplets_anchor_positive_margin1_'+str(tam_train)+'_'+str(tam_val)+'.h5', t)


	if t == 0:
		# network definition
		network = build_network([len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])])
	elif t == 1:
		# network definition
		network = build_network_deep1([len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])])
	elif t == 2:
		# network definition
		network = build_network_deep2([len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])])
	else:
		# network definition
		network = build_network_deep3([len(positive_places_train[0]), len(positive_imagenet_train[0]), len(positive_reID_train[0])])	



	path_model = 'best_modelo_'+name_architecture+'_split_triplets_anchor_positive_margin1_'+str(tam_train)+'_'+str(tam_val)

	network.load_weights(out_path_model+'/'+path_model+'.h5')

	#----------------	
	#----------------
	#extracting features

	train_positive_data = [positive_places_train,positive_imagenet_train,positive_reID_train]
	train_negative_data = [negative_places_train,negative_imagenet_train,negative_reID_train]

	val_positive_data = [positive_places_val,positive_imagenet_val,positive_reID_val]
	val_negative_data = [negative_places_val,negative_imagenet_val,negative_reID_val]

	test_positive_data = [positive_places_test,positive_imagenet_test,positive_reID_test]
	test_negative_data = [negative_places_test,negative_imagenet_test,negative_reID_test]
	
	if dataset== 'museu_nacional' or dataset =='bangladesh_fire':
	    test_real = [places_test_real,imagenet_test_real,reID_test_real]
	
	features_positive_train = network.predict(train_positive_data)
	features_negative_train = network.predict(train_negative_data)

	features_positive_val = network.predict(val_positive_data)
	features_negative_val = network.predict(val_negative_data)

	features_positive_test = network.predict(test_positive_data)
	features_negative_test = network.predict(test_negative_data)


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

	
	#----------------
	#----------------
	#extracting and saving for the new datasets with unlabeled test images
	if dataset== 'museu_nacional' or dataset =='bangladesh_fire':
	    features_test_real = network.predict(test_real)
	    np.save(os.path.join(out_path_features,name_architecture+dataset+aug+'_test_real.npy'),features_test_real)
	

#!/usr/bin/env python3
import numpy as np
import os,sys
import glob
import random
import copy
sys.path.insert(0, '../')
from parameters_analysis.parameters_graphs import *
from projections.projections import *
from ranking_utils.ranking_construction import *
from utils.normalizations import *
from ranking_utils.ess_ranking import *


def choice_training(names_noAug, names, num_train_noAug):

	final_names = []
	final_features = []
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


def generate_ess_by_training_variation(dataset,aug,train_numbers):

	filename = '../dataset'
	dir_features = '../out_files/features'
	dir_out_features = os.path.join(dir_features, 'ess')
	if not os.path.isdir(dir_out_features):
		os.mkdir(dir_out_features)

	
	name_final_folder = dataset
	name_final_folder = name_final_folder+aug
	in_features = os.path.join(dir_features, name_final_folder)
	base_features = os.path.join(dir_features, dataset)


	#----------------	
	#----------------
	#reading features
	positive_places_train = np.load(in_features+'/positivetrain_places.npy')
	positive_places_val = np.load(in_features+'/positiveval_places.npy')
	positive_places_test = np.load(in_features+'/positivetest_places.npy')
	positive_imagenet_train = np.load(in_features+'/positivetrain_imagenet.npy')
	positive_imagenet_val = np.load(in_features+'/positiveval_imagenet.npy')
	positive_imagenet_test = np.load(in_features+'/positivetest_imagenet.npy')
	positive_reID_train = np.load(in_features+'/positivetrain_reID.npy')
	positive_reID_val = np.load(in_features+'/positiveval_reID.npy')
	positive_reID_test = np.load(in_features+'/positivetest_reID.npy')

	negative_places_train = np.load(in_features+'/negativetrain_places.npy')
	negative_places_val = np.load(in_features+'/negativeval_places.npy')
	negative_places_test = np.load(in_features+'/negativetest_places.npy')
	negative_imagenet_train = np.load(in_features+'/negativetrain_imagenet.npy')
	negative_imagenet_val = np.load(in_features+'/negativeval_imagenet.npy')
	negative_imagenet_test = np.load(in_features+'/negativetest_imagenet.npy')
	negative_reID_train = np.load(in_features+'/negativetrain_reID.npy')
	negative_reID_val = np.load(in_features+'/negativeval_reID.npy')
	negative_reID_test = np.load(in_features+'/negativetest_reID.npy')


	people= np.concatenate([positive_reID_test, negative_reID_test], axis=0)
	people_z = normZ(people)
	places= np.concatenate([positive_places_test, negative_places_test], axis=0)
	places_z = normZ(places)
	imagenet= np.concatenate([positive_imagenet_test, negative_imagenet_test], axis=0)
	imagenet_z = normZ(imagenet)
	test_where = np.concatenate([imagenet_z, places_z], axis=1)
	complete = np.concatenate([places_z, imagenet_z], axis=1)
	complete = np.concatenate([complete, people_z], axis=1)

	train_names_noAug = np.load(base_features+'/positivetrain_names_imagenet.npy')
	train_names_aug = np.load(in_features+'/positivetrain_names_imagenet.npy')

	val_names_noAug = np.load(base_features+'/positiveval_names_imagenet.npy')
	val_names_aug = np.load(in_features+'/positiveval_names_imagenet.npy')

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

	out_path_features = os.path.join(dir_out_features, name_final_folder)
	if not os.path.isdir(out_path_features):
		os.mkdir(out_path_features)

	np.save(os.path.join(out_path_features,'train_indexes_'+dataset+aug+'.npy'),train_samples)
	np.save(os.path.join(out_path_features,'val_indexes_'+dataset+aug+'.npy'),val_samples)

	for j in range(iterations):
		print('------------------')
		print('------------------')
		print('Iterations:', i)

		for num_train in train_numbers:
			if num_train < len(train_names_noAug):
				print('------------------')
				print('------------------')
				print('Num train:', num_train)
				if not os.path.isfile(os.path.join(out_path_features,str(j)+'_ess_train_positive_'+str(num_train)+'.npy')):
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

				
					train_names, index_positive_train = choice_training(train_names_noAug, train_names_aug, train_samples[j][0:num_train])

					train_places_z = normZ(positive_places_train[index_positive_train])
					train_imagenet_z = normZ(positive_imagenet_train[index_positive_train])
					train_reID_z = normZ(positive_reID_train[index_positive_train])

					train_where = np.concatenate([train_imagenet_z, train_places_z], axis=1)		

					train_complete = np.concatenate([train_places_z, train_imagenet_z], axis=1)
					train_complete = np.concatenate([train_complete, train_reID_z], axis=1)
				
					
					query = copy.copy(train_where)
					distancias_where = pairwise_distances(test_where, Y=query, metric='euclidean')
					distancias_where_query = pairwise_distances(train_where, metric='euclidean')

					query= copy.copy(train_imagenet_z)
					distancias_objects = pairwise_distances(imagenet_z, Y=query, metric='euclidean')
					distancias_objects_query = pairwise_distances(train_imagenet_z, metric='euclidean')
					
					query= copy.copy(train_reID_z)
					distancias_people = pairwise_distances(people_z, Y=query, metric='euclidean')
					distancias_people_query = pairwise_distances(train_reID_z, metric='euclidean')

					#----------------	
					#----------------
					#extracting features
					description_ess = create_ess_representation([distancias_where,distancias_objects,distancias_people], 3, range(len(train_where)))
					description_ess_query = create_ess_representation([distancias_where_query,distancias_objects_query,distancias_people_query], 3, range(len(train_where)))

					#----------------	
					#----------------
					#saving extracted features
					np.save(os.path.join(out_path_features,str(j)+'_ess_test_'+str(num_train)+'.npy'),description_ess)
					np.save(os.path.join(out_path_features,str(j)+'_ess_train_positive_'+str(num_train)+'.npy'),description_ess_query)
				else:
					print('Already extracted:', j)

				
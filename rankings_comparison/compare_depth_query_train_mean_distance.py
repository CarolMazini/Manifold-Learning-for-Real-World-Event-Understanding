#!/usr/bin/env python3
import numpy as np
import os,sys,copy
import glob
import random
sys.path.insert(0, '../')
from parameters_analysis.parameters_graphs import *
from projections.projections import *
from ranking_utils.ranking_construction import *
from utils.read_dataframe import *
from utils.normalizations import *
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import argparse


##################################MAIN#####################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compare training architectures using a specific loss.')
	parser.add_argument('--dataset', dest='dataset', type=str,help='include the name of the dataset to extract image features',default='bombing')
	parser.add_argument('--aug', dest='aug', type=str,help='include _aug to use augmented data', default='')
	parser.add_argument('--method', dest='method', type=str,help='compare method used for combination', default='triplet')
	parser.add_argument("--arch0", action='store_true', help="compare architecture 512_128")
	parser.add_argument("--arch1", action='store_true', help="compare architecture 512_128_56")
	parser.add_argument("--arch2", action='store_true', help="compare architecture 1024_512")
	parser.add_argument("--arch3", action='store_true', help="compare architecture 1024_512_128")
	args = parser.parse_args()

	
	dataset = args.dataset #'wedding', 'fire', 'bombing', 'museu_nacional' or 'bangladesh_fire'
	aug=args.aug
	type_loss = args.method


	compare_arch0 = args.arch0
	compare_arch1 = args.arch1
	compare_arch2 = args.arch2
	compare_arch3 = args.arch3

	

	path_base = '../out_files/features'
	path_base_ranking = '../out_files/ranking/'+type_loss
	path_base_graphs = '../out_files/graphs'
	
	name_final_folder = dataset+aug

	save_path_graph = os.path.join(path_base_graphs,type_loss)
	if not os.path.isdir(save_path_graph):
		os.mkdir(save_path_graph)

	save_path_graph = os.path.join(save_path_graph,name_final_folder)
	if not os.path.isdir(save_path_graph):
		os.mkdir(save_path_graph)

	## feature paths
	path_features_deep0 = os.path.join(path_base,type_loss,name_final_folder)

	## ranking paths

	path_base_ranking = os.path.join(path_base_ranking,name_final_folder)
	if not os.path.isdir(path_base_ranking):
		os.mkdir(path_base_ranking)

	path_ranking_arch0 = os.path.join(path_base_ranking,'arch0')
	if not os.path.isdir(path_ranking_arch0):
		os.mkdir(path_ranking_arch0)
	path_ranking_arch0 = os.path.join(path_ranking_arch0,name_final_folder)
	if not os.path.isdir(path_ranking_arch0):
		os.mkdir(path_ranking_arch0)

	path_ranking_arch1 = os.path.join(path_base_ranking,'arch1')
	if not os.path.isdir(path_ranking_arch1):
		os.mkdir(path_ranking_arch1)
	path_ranking_arch1 = os.path.join(path_ranking_arch1,name_final_folder)
	if not os.path.isdir(path_ranking_arch1):
		os.mkdir(path_ranking_arch1)

	path_ranking_arch2 = os.path.join(path_base_ranking,'arch2')
	if not os.path.isdir(path_ranking_arch2):
		os.mkdir(path_ranking_arch2)
	path_ranking_arch2 = os.path.join(path_ranking_arch2,name_final_folder)
	if not os.path.isdir(path_ranking_arch2):
		os.mkdir(path_ranking_arch2)

	path_ranking_arch3 = os.path.join(path_base_ranking,'arch3')
	if not os.path.isdir(path_ranking_arch3):
		os.mkdir(path_ranking_arch3)
	path_ranking_arch3 = os.path.join(path_ranking_arch3,name_final_folder)
	if not os.path.isdir(path_ranking_arch3):
		os.mkdir(path_ranking_arch3)

	##------------------------
	##------------------------
	## Load features

	#### arch0
	if compare_arch0:
		c_p_test = np.load(os.path.join(path_features_deep0,'512_128_positive_'+dataset+aug+'_test.npy'))
		c_n_test = np.load(os.path.join(path_features_deep0,'512_128_negative_'+dataset+aug+'_test.npy'))
		deep0_complete = np.concatenate([c_p_test, c_n_test], axis=0)
		deep0_train_complete = np.load(os.path.join(path_features_deep0,'512_128_positive_'+dataset+aug+'_train.npy'))
		notSortedIndicesDeep0 = []
		sortedIndicesDeep0 = []
		mean_deep0 = []
		var_deep0 = []
		non_mean_deep0 = []
		non_var_deep0 = []

	#### arch1
	if compare_arch1:
		cont_p_test = np.load(os.path.join(path_features_deep0,'512_128_64_positive_'+dataset+aug+'_test.npy'))
		cont_n_test = np.load(os.path.join(path_features_deep0,'512_128_64_negative_'+dataset+aug+'_test.npy'))
		deep1_complete = np.concatenate([cont_p_test, cont_n_test], axis=0)
		deep1_train_complete = np.load(os.path.join(path_features_deep0,'512_128_64_positive_'+dataset+aug+'_train.npy'))
		notSortedIndicesDeep1 = []
		sortedIndicesDeep1 = []
		mean_deep1 = []
		var_deep1 = []
		non_mean_deep1 = []
		non_var_deep1 = []

	#### arch2
	if compare_arch2:
		t_p_test = np.load(os.path.join(path_features_deep0,'1024_512_positive_'+dataset+aug+'_test.npy'))
		t_n_test = np.load(os.path.join(path_features_deep0,'1024_512_negative_'+dataset+aug+'_test.npy'))
		deep2_complete = np.concatenate([t_p_test, t_n_test], axis=0)
		deep2_train_complete = np.load(os.path.join(path_features_deep0,'1024_512_positive_'+dataset+aug+'_train.npy'))
		notSortedIndicesDeep2 = []
		sortedIndicesDeep2 = []
		mean_deep2 = []
		var_deep2 = []
		non_mean_deep2 = []
		non_var_deep2 = []

	#### arch3
	if compare_arch3:
		t_p_test = np.load(os.path.join(path_features_deep0,'1024_512_128_positive_'+dataset+aug+'_test.npy'))
		t_n_test = np.load(os.path.join(path_features_deep0,'1024_512_128_negative_'+dataset+aug+'_test.npy'))
		deep3_complete = np.concatenate([t_p_test, t_n_test], axis=0)
		deep3_train_complete = np.load(os.path.join(path_features_deep0,'1024_512_128_positive_'+dataset+aug+'_train.npy'))
		notSortedIndicesDeep3 = []
		sortedIndicesDeep3 = []
		mean_deep3 = []
		var_deep3 = []
		non_mean_deep3 = []
		non_var_deep3 = []


	relevant = len(t_p_test)
	iterations = len(deep2_train_complete)
	print('Iterations:', iterations)
	#iterations = 10
	count_normal = 0
	count_embedding = 0
	relevant = 2 


	##------------------------
	##------------------------
	## Obtain one ranking for query (positive images of training)

	for iteration in range(0,iterations):
		print('--------------------')
		print('Iteration',iteration)

		if compare_arch0:

			query = []
			query.append(np.copy(deep0_train_complete[int(iteration)]))
			distancias_deep0 = pairwise_distances(deep0_complete, Y=query, metric='euclidean')
			mean_deep0.append(np.mean(distancias_deep0[0:relevant]))
			non_mean_deep0.append(np.mean(distancias_deep0[relevant:-1]))
			var_deep0.append(np.var(distancias_deep0[0:relevant]))
			non_var_deep0.append(np.var(distancias_deep0[relevant:-1]))
			rankingDeep0 = np.transpose(distancias_deep0)
			notSortedIndicesDeep0.append(rankingDeep0[0])
			sortedIndicesDeep0.append(np.argsort(rankingDeep0)[0])

		if compare_arch1:

			query = []
			query.append(np.copy(deep1_train_complete[int(iteration)]))
			distancias_deep1 = pairwise_distances(deep1_complete, Y=query, metric='euclidean')
			mean_deep1.append(np.mean(distancias_deep1[0:relevant]))
			non_mean_deep1.append(np.mean(distancias_deep1[relevant:-1]))
			var_deep1.append(np.var(distancias_deep1[0:relevant]))
			non_var_deep1.append(np.var(distancias_deep1[relevant:-1]))
			rankingDeep1 = np.transpose(distancias_deep1)
			notSortedIndicesDeep1.append(rankingDeep1[0])
			sortedIndicesDeep1.append(np.argsort(rankingDeep1)[0])

		if compare_arch2:

			query = []
			query.append(np.copy(deep2_train_complete[int(iteration)]))
			distancias_deep2 = pairwise_distances(deep2_complete, Y=query, metric='euclidean')
			mean_deep2.append(np.mean(distancias_deep2[0:relevant]))
			non_mean_deep2.append(np.mean(distancias_deep2[relevant:-1]))
			var_deep2.append(np.var(distancias_deep2[0:relevant]))
			non_var_deep2.append(np.var(distancias_deep2[relevant:-1]))
			rankingDeep2 = np.transpose(distancias_deep2)
			notSortedIndicesDeep2.append(rankingDeep2[0])
			sortedIndicesDeep2.append(np.argsort(rankingDeep2)[0])
		
		if compare_arch3:

			query = []
			query.append(np.copy(deep3_train_complete[int(iteration)]))
			distancias_deep3 = pairwise_distances(deep3_complete, Y=query, metric='euclidean')
			mean_deep3.append(np.mean(distancias_deep3[0:relevant]))
			non_mean_deep3.append(np.mean(distancias_deep3[relevant:-1]))
			var_deep3.append(np.var(distancias_deep3[0:relevant]))
			non_var_deep3.append(np.var(distancias_deep3[relevant:-1]))
			rankingDeep3 = np.transpose(distancias_deep3)
			notSortedIndicesDeep3.append(rankingDeep3[0])
			sortedIndicesDeep3.append(np.argsort(rankingDeep3)[0])

	##------------------------
	##------------------------
	## Save and analyse rankings

	media_final = []
	var_final = []
	all_mean = []
	all_var = []
	all_mean_not = []
	all_var_not = []
	subtitle = []

	if compare_arch0:
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesDeep0,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		np.save(os.path.join(path_ranking_arch0,'distances_deep0_'+dataset+'_'+aug+'.npy'), np.array(notSortedIndicesDeep0))
		np.save(os.path.join(path_ranking_arch0,'rankings_relevant_deep0_'+dataset+aug+'.npy'),sortedIndicesDeep0)
		subtitle.append('512_128')
		all_mean.append(np.mean(mean_deep0))
		all_var.append(np.mean(var_deep0))
		all_mean_not.append(np.mean(non_mean_deep0))
		all_var_not.append(np.mean(non_var_deep0))

	if compare_arch1:
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesDeep1,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		np.save(os.path.join(path_ranking_arch1,'distances_deep1_'+dataset+'_'+aug+'.npy'), np.array(notSortedIndicesDeep1))
		np.save(os.path.join(path_ranking_arch1,'rankings_relevant_deep1_'+dataset+aug+'.npy'),sortedIndicesDeep1)
		subtitle.append('512_128_64')
		all_mean.append(np.mean(mean_deep1))
		all_var.append(np.mean(var_deep1))
		all_mean_not.append(np.mean(non_mean_deep1))
		all_var_not.append(np.mean(non_var_deep1))

	if compare_arch2:
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesDeep2,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		np.save(os.path.join(path_ranking_arch2,'distances_deep2_'+dataset+'_'+aug+'.npy'), np.array(notSortedIndicesDeep2))
		np.save(os.path.join(path_ranking_arch2,'rankings_relevant_deep2_'+dataset+aug+'.npy'),sortedIndicesDeep2)
		subtitle.append('1024_512')
		all_mean.append(np.mean(mean_deep2))
		all_var.append(np.mean(var_deep2))
		all_mean_not.append(np.mean(non_mean_deep2))
		all_var_not.append(np.mean(non_var_deep2))

	if compare_arch3:
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesDeep3,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		np.save(os.path.join(path_ranking_arch3,'distances_deep3_'+dataset+'_'+aug+'.npy'), np.array(notSortedIndicesDeep3))
		np.save(os.path.join(path_ranking_arch3,'rankings_relevant_deep3_'+dataset+aug+'.npy'),sortedIndicesDeep3)
		subtitle.append('1024_512_128')
		all_mean.append(np.mean(mean_deep3))
		all_var.append(np.mean(var_deep3))
		all_mean_not.append(np.mean(non_mean_deep3))
		all_var_not.append(np.mean(non_var_deep3))


	## save mean distances and variance
	np.save(os.path.join(path_base_ranking,'deep_positive_mean_distances_'+dataset+'_'+aug+'.npy'), np.array(all_mean))
	np.save(os.path.join(path_base_ranking,'deep_negative_mean_distances_'+dataset+'_'+aug+'.npy'), np.array(all_mean_not))
	np.save(os.path.join(path_base_ranking,'deep_positive_var_distances_'+dataset+'_'+aug+'.npy'), np.array(all_var))
	np.save(os.path.join(path_base_ranking,'deep_negative_var_distances_'+dataset+'_'+aug+'.npy'), np.array(all_var_not))
	

	## save values for graph
	np.save(os.path.join(path_base_ranking,'deep_points_mean_graph.npy'), np.array(media_final))
	np.save(os.path.join(path_base_ranking,'deep_points_var_graph.npy'), np.array(var_final))

	## plot comparison
	cores = [ '#d8b989','#62c9bf', '#746a9e', '#e47c7b','#91c4f2']
	mark_precision = ['o','<','*', '>', 'X', 'D']
	line_error_plot(media_final,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],var_final, len(subtitle), mark_precision,'Recall','Precision', subtitle,os.path.join(save_path_graph,'precision_recall_comparison_depth.pdf'),cores,dataset)


	

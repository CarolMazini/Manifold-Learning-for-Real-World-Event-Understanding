#!/usr/bin/env python3
import numpy as np
import random
import os, sys
import shutil
from generate_ess_for_vary_comparison import *
sys.path.insert(0, '../')
from ranking_utils.rank_metrics import *
from parameters_analysis.parameters_graphs import *
from projections.projections import *
from ranking_utils.ranking_construction import *
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import argparse

##################################MAIN#####################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compare different size training sets. The code uses augmented data.')
	parser.add_argument('--dataset', dest='dataset', type=str,help='include the name of the dataset to extract image features',default='bombing')
	#parser.add_argument('--aug', dest='aug', type=str,help='include _aug to use augmented data', default='_aug')
	parser.add_argument('--method', dest='method', type=str,help='compare method used for combination', default='triplet')
	args = parser.parse_args()

	
	d = args.dataset #'wedding', 'fire', 'bombing', 'museu_nacional' or 'bangladesh_fire'
	a='_aug'
	t = args.method

	train_numbers = [10,20,50,100,200]
	iterations = 10

	filename = '../dataset'
	dir_features = '../out_files/features'
	path_base_graphs = '../out_files/graphs'


	save_path_graph = os.path.join(path_base_graphs,t)
	if not os.path.isdir(save_path_graph):
		os.mkdir(save_path_graph)
	
		
	##generate the ESS rankings
	generate_ess_by_training_variation(d,a,train_numbers)

	name_final_folder = d+a
	path_features_networks = os.path.join(dir_features, t,name_final_folder)
	path_features_ess = os.path.join(dir_features, 'ess',name_final_folder)
	save_path_graph = os.path.join(save_path_graph,name_final_folder)
	if not os.path.isdir(save_path_graph):
		os.mkdir(save_path_graph)

	val_names_noAug = np.load(os.path.join(dir_features,d,'positive_val_names.npy'))
	
	aux = 0
	map_final = []
	media_final = []
	var_final = []
	var_map_final = []
	var_final_512 = []
	map_final_512 = []
	media_final_512 = []
	var_map_final_512 = []
	var_final_1024 = []
	map_final_1024 = []
	media_final_1024 = []
	var_map_final_1024 = []
	map_final_ess = []
	media_final_ess = []
	var_final_ess = []
	var_map_final_ess = []


	for num in range(len(train_numbers)):
		num_val = int(train_numbers[num]*0.2)
		if num_val>len(val_names_noAug):
			num_val =len(val_names_noAug)
		media_512 = []
		var_512=[]
		map_512 = []
		media_1024 = []
		var_1024=[]
		map_1024 = []
		media_ess = []
		var_ess=[]
		map_ess = []
		for iterat in range(iterations):

			##------------------------
			##------------------------
			## Load features

			##512_128
			train_complete_512 = np.load(os.path.join(path_features_networks,str(iterat)+'_positive_'+str(train_numbers[num])+'_'+str(num_val)+name_final_folder+'_train_512_128.npy'))
			test_positive_512 = np.load(os.path.join(path_features_networks,str(iterat)+'_positive_'+str(train_numbers[num])+'_'+str(num_val)+name_final_folder+'_test_512_128.npy'))
			test_negative_512 = np.load(os.path.join(path_features_networks,str(iterat)+'_negative_'+str(train_numbers[num])+'_'+str(num_val)+name_final_folder+'_test_512_128.npy'))
			complete_512 = np.concatenate([test_positive_512, test_negative_512], axis=0)

			##1024_512
			train_complete_1024 = np.load(os.path.join(path_features_networks,str(iterat)+'_positive_'+str(train_numbers[num])+'_'+str(num_val)+name_final_folder+'_train_1024_512.npy'))
			test_positive_1024 = np.load(os.path.join(path_features_networks,str(iterat)+'_positive_'+str(train_numbers[num])+'_'+str(num_val)+name_final_folder+'_test_1024_512.npy'))
			test_negative_1024 = np.load(os.path.join(path_features_networks,str(iterat)+'_negative_'+str(train_numbers[num])+'_'+str(num_val)+name_final_folder+'_test_1024_512.npy'))
			complete_1024 = np.concatenate([test_positive_1024, test_negative_1024], axis=0)


			##ess
			train_complete_ess = np.load(os.path.join(path_features_ess,str(iterat)+'_ess_train_positive_'+str(train_numbers[num])+'.npy'))
			test_ess = np.load(os.path.join(path_features_ess,str(iterat)+'_ess_test_'+str(train_numbers[num])+'.npy'))

			tam_train = len(train_complete_ess)
			relevant = len(test_positive_1024)
			sortedIndices512 = []
			sortedIndices1024 = []
			sortedIndicesESS = []

			##------------------------
			##------------------------
			## Obtain rankings

			for i in range(tam_train):

				#512_128
				query = []
				query.append(np.copy(train_complete_512[i]))
				distancias_512 = pairwise_distances(complete_512, Y=query, metric='euclidean')
				ranking_512= np.transpose(distancias_512)
				sortedIndices512.append(np.argsort(ranking_512)[0])

				#1024_512
				query = []
				query.append(np.copy(train_complete_1024[i]))
				distancias_1024 = pairwise_distances(complete_1024, Y=query, metric='euclidean')
				ranking_1024= np.transpose(distancias_1024)
				sortedIndices1024.append(np.argsort(ranking_1024)[0])

				#ESS
				query = []
				query.append(np.copy(train_complete_ess[i]))
				distancias_ess = pairwise_distances(test_ess, Y=query, metric='euclidean')
				ranking_ess= np.transpose(distancias_ess)
				sortedIndicesESS.append(np.argsort(ranking_ess)[0])

			precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndices512,[],np.arange(0,relevant))
			map_512.append(map_values[0])
			media_512.append(np.copy(media))
			var_512.append(np.copy(var))

			precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndices1024,[],np.arange(0,relevant))
			map_1024.append(map_values[0])
			media_1024.append(np.copy(media))
			var_1024.append(np.copy(var))

			precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesESS,[],np.arange(0,relevant))
			map_ess.append(map_values[0])
			media_ess.append(np.copy(media))
			var_ess.append(np.copy(var))


		##------------------------
		##------------------------
		## Rankings mean and variance

		map_final_512.append(np.mean(np.array(map_512)))
		var_map_final_512.append(np.var(np.array(map_512)))
		media_final_512.append(np.mean(np.array(media_512), axis=0))
		var_final_512.append(np.mean(np.array(var_512), axis=0))

		map_final_1024.append(np.mean(np.array(map_1024)))
		var_map_final_1024.append(np.var(np.array(map_1024)))
		media_final_1024.append(np.mean(np.array(media_1024), axis=0))
		var_final_1024.append(np.mean(np.array(var_1024), axis=0))

		map_final_ess.append(np.mean(np.array(map_ess)))
		var_map_final_ess.append(np.var(np.array(map_ess)))
		media_final_ess.append(np.mean(np.array(media_ess), axis=0))
		var_final_ess.append(np.mean(np.array(var_ess), axis=0))
		aux+=1

			
	
	print('Map:',map_final)
	var_map_final.append(np.array(var_map_final_512))
	var_map_final.append(np.array(var_map_final_1024))
	var_map_final.append(np.array(var_map_final_ess))

	print(var_map_final)

	map_final.append(np.array(map_final_512))
	map_final.append(np.array(map_final_1024))
	map_final.append(np.array(map_final_ess))

	print(map_final)

	media_final.append(np.array(media_final_512))
	media_final.append(np.array(media_final_1024))
	media_final.append(np.array(media_final_ess))

	var_final.append(np.array(var_final_512))
	var_final.append(np.array(var_final_1024))
	var_final.append(np.array(var_final_ess))

	##------------------------
	##------------------------
	## Save for plot

	np.save(os.path.join(save_path_graph,'all_mean_map_'+d+'_'+t+a+'_'+str(train_numbers[num])+'.npy'), map_final)
	np.save(os.path.join(save_path_graph,'all_var_map_'+d+'_'+t+a+'_'+str(train_numbers[num])+'.npy'), var_map_final)
	np.save(os.path.join(save_path_graph,'all_mean_points_mean_graph_'+d+'_'+t+a+'_'+str(train_numbers[num])+'.npy'), np.array(media_final))
	np.save(os.path.join(save_path_graph,'all_mean_points_var_graph_'+d+'_'+t+a+'_'+str(train_numbers[num])+'.npy'), np.array(var_final))

	y_values = np.array(train_numbers)[0:aux]
	cores = ['#726a9e', '#ad7a99', '#d8b989']
	mark_precision = ['o','<','*']
	line_error_plot(map_final,y_values,var_map_final, 2, mark_precision,'Positive Training','MAP', ['512_128', '1024_512', 'ESS'],os.path.join(save_path_graph,'comparision_vary_train_all_mean_map_'+d+'_'+t+a+'_'+str(train_numbers[num])+'.pdf'),cores,d)




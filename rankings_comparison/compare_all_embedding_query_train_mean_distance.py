#!/usr/bin/env python3
import numpy as np
import os,sys,copy
import glob
import random
sys.path.insert(0, '../')
from parameters_analysis.parameters_graphs import *
from projections.projections import *
from ranking_utils.ranking_construction import *
from utils.normalizations import *
from ranking_utils.ess_ranking import *
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import argparse





def load_features(base, names):
	p_pl_train = np.load(os.path.join(base,names[0]))
	p_pl_val = np.load(os.path.join(base,names[1]))
	p_pl_test = np.load(os.path.join(base,names[2]))
	p_im_train = np.load(os.path.join(base,names[3]))
	p_im_val = np.load(os.path.join(base,names[4]))
	p_im_test = np.load(os.path.join(base,names[5]))
	p_reID_train = np.load(os.path.join(base,names[6]))
	p_reID_val = np.load(os.path.join(base,names[7]))
	p_reID_test = np.load(os.path.join(base,names[8]))

	n_pl_train = np.load(os.path.join(base,names[9]))
	n_pl_val = np.load(os.path.join(base,names[10]))
	n_pl_test = np.load(os.path.join(base,names[11]))
	n_im_train = np.load(os.path.join(base,names[12]))
	n_im_val = np.load(os.path.join(base,names[13]))
	n_im_test = np.load(os.path.join(base,names[14]))
	n_reID_train = np.load(os.path.join(base,names[15]))
	n_reID_val = np.load(os.path.join(base,names[16]))
	n_reID_test = np.load(os.path.join(base,names[17]))

	return [p_pl_train,p_pl_val,p_pl_test],[p_im_train,p_im_val,p_im_test],[p_reID_train,p_reID_val,p_reID_test],[n_pl_train,n_pl_val,n_pl_test],[n_im_train,n_im_val,n_im_test],[n_reID_train,n_reID_val,n_reID_test]


##################################MAIN#####################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compare different methods using best architecture.')
	parser.add_argument('--dataset', dest='dataset', type=str,help='include the name of the dataset to extract image features',default='bombing')
	parser.add_argument('--aug', dest='aug', type=str,help='include _aug to use augmented data', default='')
	#parser.add_argument("--concatenation", action='store_true', help="compare concatenation")
	parser.add_argument("--finetuning", action='store_true', help="compare concatenation of fine-tuned features")
	parser.add_argument("--ESS", action='store_true', help="compare ESS features")
	parser.add_argument("--cross_entropy", action='store_true', help="compare cross_entropy features")
	parser.add_argument("--contrastive", action='store_true', help="compare contrastive features")
	parser.add_argument("--triplet", action='store_true',help="compare triplet features")
	args = parser.parse_args()

	
	dataset = args.dataset #'wedding', 'fire', 'bombing', 'museu_nacional' or 'bangladesh_fire'
	aug=args.aug


	compare_concat = True
	compare_finetuned = args.finetuning
	compare_ESS = args.ESS
	compare_cross = args.cross_entropy
	compare_contrastive = args.contrastive
	compare_triplet = args.triplet


	path_base = '../out_files/features'
	path_base_ranking = '../out_files/ranking'
	path_base_graphs = '../out_files/graphs'
	
	name_final_folder = dataset+aug

	save_path_graph = os.path.join(path_base_graphs,name_final_folder)
	if not os.path.isdir(save_path_graph):
		os.mkdir(save_path_graph)

	## feature paths
	path_features = os.path.join(path_base,name_final_folder)
	path_features_finetuning = os.path.join(path_base,'fine_tunned',name_final_folder)
	path_features_crossEntropy = os.path.join(path_base,'cross_entropy',name_final_folder)
	path_features_contrastive = os.path.join(path_base,'contrastive',name_final_folder)
	path_features_triplet = os.path.join(path_base,'triplet',name_final_folder)
	path_features_ess = os.path.join(path_base,'ess')
	if not os.path.isdir(path_features_ess):
		os.mkdir(path_features_ess)
	path_features_ess = os.path.join(path_features_ess, name_final_folder)
	if not os.path.isdir(path_features_ess):
		os.mkdir(path_features_ess)

	## ranking paths
	path_ranking = os.path.join(path_base_ranking,name_final_folder)
	if not os.path.isdir(path_ranking):
		os.mkdir(path_ranking)

	path_ranking_finetuning = os.path.join(path_base_ranking,'fine_tunned')
	if not os.path.isdir(path_ranking_finetuning):
		os.mkdir(path_ranking_finetuning)
	path_ranking_finetuning = os.path.join(path_ranking_finetuning,name_final_folder)
	if not os.path.isdir(path_ranking_finetuning):
		os.mkdir(path_ranking_finetuning)

	path_ranking_crossEntropy = os.path.join(path_base_ranking,'cross_entropy')
	if not os.path.isdir(path_ranking_crossEntropy):
		os.mkdir(path_ranking_crossEntropy)
	path_ranking_crossEntropy = os.path.join(path_ranking_crossEntropy,name_final_folder)
	if not os.path.isdir(path_ranking_crossEntropy):
		os.mkdir(path_ranking_crossEntropy)

	path_ranking_contrastive = os.path.join(path_base_ranking,'contrastive')
	if not os.path.isdir(path_ranking_contrastive):
		os.mkdir(path_ranking_contrastive)
	path_ranking_contrastive = os.path.join(path_ranking_contrastive,name_final_folder)
	if not os.path.isdir(path_ranking_contrastive):
		os.mkdir(path_ranking_contrastive)

	path_ranking_triplet = os.path.join(path_base_ranking,'triplet')
	if not os.path.isdir(path_ranking_triplet):
		os.mkdir(path_ranking_triplet)
	path_ranking_triplet = os.path.join(path_ranking_triplet,name_final_folder)
	if not os.path.isdir(path_ranking_triplet):
		os.mkdir(path_ranking_triplet)

	path_ranking_ess = os.path.join(path_base_ranking,'ess')
	if not os.path.isdir(path_ranking_ess):
		os.mkdir(path_ranking_ess)
	path_ranking_ess = os.path.join(path_ranking_ess, name_final_folder)
	if not os.path.isdir(path_ranking_ess):
		os.mkdir(path_ranking_ess)

	##------------------------
	##------------------------
	## Load features
	names_normal = ['positive_train_places.npy','positive_val_places.npy','positive_test_places.npy','positive_train_imagenet.npy','positive_val_imagenet.npy',
					'positive_test_imagenet.npy','positive_train_people.npy','positive_val_people.npy','positive_test_people.npy','negative_train_places.npy','negative_val_places.npy',
					'negative_test_places.npy','negative_train_imagenet.npy','negative_val_imagenet.npy','negative_test_imagenet.npy','negative_train_people.npy','negative_val_people.npy',
					'negative_test_people.npy']


	###normal
	if compare_concat:
		p_pl, p_im, p_reID, n_pl, n_im, n_reID=load_features(path_features, names_normal)

		people= np.concatenate([p_reID[2], n_reID[2]], axis=0)
		people_z = normZ(people)
		places= np.concatenate([p_pl[2], n_pl[2]], axis=0)
		places_z = normZ(places)
		imagenet= np.concatenate([p_im[2], n_im[2]], axis=0)
		imagenet_z = normZ(imagenet)

		test_where = np.concatenate([imagenet_z, places_z], axis=1)

		complete = np.concatenate([places_z, imagenet_z], axis=1)
		complete = np.concatenate([complete, people_z], axis=1)

		train_places_z = normZ(p_pl[0])
		train_imagenet_z = normZ(p_im[0])
		train_reID_z = normZ(p_reID[0])

		train_where = np.concatenate([train_imagenet_z, train_places_z], axis=1)		

		train_complete = np.concatenate([train_places_z, train_imagenet_z], axis=1)
		train_complete = np.concatenate([train_complete, train_reID_z], axis=1)

	
		notSortedIndicesConcat = []
		sortedIndicesConcat = []
		mean_concat = []
		var_concat = []
		non_mean_concat = []
		non_var_concat = []
	

	###finetuned
	if(compare_finetuned):
		f_p_pl, f_p_im, f_p_reID, f_n_pl, f_n_im, f_n_reID=load_features(path_features_finetuning, names_normal)


		finetuning_people= np.concatenate([f_p_reID[2], f_n_reID[2]], axis=0)
		finetuning_people_z = normZ(finetuning_people)
		finetuning_places=  np.concatenate([f_p_pl[2], f_n_pl[2]], axis=0)
		finetuning_places_z = normZ(finetuning_places)
		finetuning_imagenet= np.concatenate([f_p_im[2], f_n_im[2]], axis=0)
		finetuning_imagenet_z = normZ(finetuning_imagenet)

		finetuning_complete = np.concatenate([finetuning_places_z, finetuning_imagenet_z], axis=1)
		finetuning_complete = np.concatenate([finetuning_complete, finetuning_people_z], axis=1)



		f_train_places_z = normZ(f_p_pl[0])
		f_train_imagenet_z = normZ(f_p_im[0])
		f_train_reID_z = normZ(f_p_reID[0])

		finetuning_train_complete = np.concatenate([f_train_places_z, f_train_imagenet_z], axis=1)
		finetuning_train_complete = np.concatenate([finetuning_train_complete, f_train_reID_z], axis=1)
	
		notSortedIndicesFinetuning = []
		sortedIndicesFinetuning = []
		mean_fine = []
		var_fine = []
		non_mean_fine = []
		non_var_fine = []
	
	####ess
	if compare_ESS:
		try:
			description_ess = np.load(os.path.join(path_features_ess,'ess_test.npy'))
			description_ess_query = np.load(os.path.join(path_features_ess,'ess_train.npy'))

		except:

			query = copy.copy(train_where)
			distancias_where = pairwise_distances(test_where, Y=query, metric='euclidean')
			distancias_where_query = pairwise_distances(train_where, metric='euclidean')

			query= copy.copy(train_imagenet_z)
			distancias_objects = pairwise_distances(imagenet_z, Y=query, metric='euclidean')
			distancias_objects_query = pairwise_distances(train_imagenet_z, metric='euclidean')
			
			query= copy.copy(train_reID_z)
			distancias_people = pairwise_distances(people_z, Y=query, metric='euclidean')
			distancias_people_query = pairwise_distances(train_reID_z, metric='euclidean')

			description_ess = create_ess_representation([distancias_where,distancias_objects,distancias_people], 3, range(len(train_where)))
			description_ess_query = create_ess_representation([distancias_where_query,distancias_objects_query,distancias_people_query], 3, range(len(train_where)))

			np.save(os.path.join(path_features_ess,'ess_test.npy'),description_ess)
			np.save(os.path.join(path_features_ess,'ess_train.npy'),description_ess_query)

		notSortedIndicesESS = []
		sortedIndicesESS = []
		mean_ESS = []
		var_ESS = []
		non_mean_ESS = []
		non_var_ESS = []

	

	####cross entropy
	if compare_cross:
		c_p_test = np.load(os.path.join(path_features_crossEntropy,'1024_512_positive_'+dataset+aug+'_test.npy'))
		c_n_test = np.load(os.path.join(path_features_crossEntropy,'1024_512_negative_'+dataset+aug+'_test.npy'))
		cross_complete = np.concatenate([c_p_test, c_n_test], axis=0)
		cross_train_complete = np.load(os.path.join(path_features_crossEntropy,'1024_512_positive_'+dataset+aug+'_train.npy'))

		notSortedIndicesCross = []
		sortedIndicesCross = []
		mean_cross = []
		var_cross = []
		non_mean_cross = []
		non_var_cross = []
			

	####contrastive
	if compare_contrastive:
		cont_p_test = np.load(os.path.join(path_features_contrastive,'1024_512_positive_'+dataset+aug+'_test.npy'))
		cont_n_test = np.load(os.path.join(path_features_contrastive,'1024_512_negative_'+dataset+aug+'_test.npy'))
		contrastive_complete = np.concatenate([cont_p_test, cont_n_test], axis=0)
		contrastive_train_complete = np.load(os.path.join(path_features_contrastive,'1024_512_positive_'+dataset+aug+'_train.npy'))

		notSortedIndicesContrastive = []
		sortedIndicesContrastive = []
		mean_cont = []
		var_cont = []
		non_mean_cont = []
		non_var_cont = []
					

	####triplet
	if compare_triplet:
		t_p_test = np.load(os.path.join(path_features_triplet,'1024_512_positive_'+dataset+aug+'_test.npy'))
		t_n_test = np.load(os.path.join(path_features_triplet,'1024_512_negative_'+dataset+aug+'_test.npy'))
		triplet_complete = np.concatenate([t_p_test, t_n_test], axis=0)
		triplet_train_complete = np.load(os.path.join(path_features_triplet,'1024_512_positive_'+dataset+aug+'_train.npy'))
	
		notSortedIndicesTriplet = []
		sortedIndicesTriplet = []
		mean_triplet = []
		var_triplet = []
		non_mean_triplet = []
		non_var_triplet = []
	

	print('Normal Lenght: ',len(complete))

	relevant = len(p_reID[2])
	
	iterations = len(p_reID[0])
	print('Iterations:', iterations)
	#iterations = len(triplet_train_complete)
	#print('Iterations:', iterations)
	#iterations = 2
	count_normal = 0
	count_embedding = 0

	##------------------------
	##------------------------
	## Obtain one ranking for query (positive images of training)

	for iteration in range(0,iterations):
		print('--------------------')
		print('Iteration',iteration)

		if compare_ESS:

			distancias_ESS = pairwise_distances(description_ess, Y=[description_ess_query[iteration]], metric='euclidean')
			
			mean_ESS.append(np.mean(distancias_ESS[0:relevant]))
			non_mean_ESS.append(np.mean(distancias_ESS[relevant:-1]))
			var_ESS.append(np.var(distancias_ESS[0:relevant]))
			non_var_ESS.append(np.var(distancias_ESS[relevant:-1]))
			rankingESS = np.transpose(distancias_ESS)
			notSortedIndicesESS.append(rankingESS[0])
			sortedIndicesESS.append(np.argsort(rankingESS)[0])

		if compare_concat:
			query = []
			query.append(np.copy(train_complete[int(iteration)]))
			distancias_concat = pairwise_distances(complete, Y=query, metric='euclidean')
			mean_concat.append(np.mean(distancias_concat[0:relevant]))
			non_mean_concat.append(np.mean(distancias_concat[relevant:-1]))
			var_concat.append(np.var(distancias_concat[0:relevant]))
			non_var_concat.append(np.var(distancias_concat[relevant:-1]))
			rankingConcat = np.transpose(distancias_concat)
			notSortedIndicesConcat.append(rankingConcat[0])
			sortedIndicesConcat.append(np.argsort(rankingConcat)[0])

		if compare_finetuned:

			query = []
			query.append(np.copy(finetuning_train_complete[int(iteration)]))
			distancias_concat_finetuning = pairwise_distances(finetuning_complete, Y=query, metric='euclidean')

			mean_fine.append(np.mean(distancias_concat_finetuning[0:relevant]))
			non_mean_fine.append(np.mean(distancias_concat_finetuning[relevant:-1]))
			var_fine.append(np.var(distancias_concat_finetuning[0:relevant]))
			non_var_fine.append(np.var(distancias_concat_finetuning[relevant:-1]))
			rankingConcatFinetuning = np.transpose(distancias_concat_finetuning)
			notSortedIndicesFinetuning.append(rankingConcatFinetuning[0])
			sortedIndicesFinetuning.append(np.argsort(rankingConcatFinetuning)[0])

		if compare_cross:

			query = []
			query.append(np.copy(cross_train_complete[int(iteration)]))
			distancias_cross = pairwise_distances(cross_complete, Y=query, metric='euclidean')
			mean_cross.append(np.mean(distancias_cross[0:relevant]))
			non_mean_cross.append(np.mean(distancias_cross[relevant:-1]))
			var_cross.append(np.var(distancias_cross[0:relevant]))
			non_var_cross.append(np.var(distancias_cross[relevant:-1]))
			rankingCross = np.transpose(distancias_cross)
			notSortedIndicesCross.append(rankingCross[0])
			sortedIndicesCross.append(np.argsort(rankingCross)[0])

		if compare_contrastive:

			query = []
			query.append(np.copy(contrastive_train_complete[int(iteration)]))
			distancias_contrastive = pairwise_distances(contrastive_complete, Y=query, metric='euclidean')
			mean_cont.append(np.mean(distancias_contrastive[0:relevant]))
			non_mean_cont.append(np.mean(distancias_contrastive[relevant:-1]))
			var_cont.append(np.var(distancias_contrastive[0:relevant]))
			non_var_cont.append(np.var(distancias_contrastive[relevant:-1]))
			rankingContrastive = np.transpose(distancias_contrastive)
			notSortedIndicesContrastive.append(rankingContrastive[0])
			sortedIndicesContrastive.append(np.argsort(rankingContrastive)[0])

		if compare_triplet:

			query = []
			query.append(np.copy(triplet_train_complete[int(iteration)]))
			distancias_triplet = pairwise_distances(triplet_complete, Y=query, metric='euclidean')
			mean_triplet.append(np.mean(distancias_triplet[0:relevant]))
			non_mean_triplet.append(np.mean(distancias_triplet[relevant:-1]))
			var_triplet.append(np.var(distancias_triplet[0:relevant]))
			non_var_triplet.append(np.var(distancias_triplet[relevant:-1]))
			rankingTriplet = np.transpose(distancias_triplet)
			notSortedIndicesTriplet.append(rankingTriplet[0])
			sortedIndicesTriplet.append(np.argsort(rankingTriplet)[0])


	##------------------------
	##------------------------
	## Save and analyse rankings
	media_final = []
	var_final = []
	cores = []
	subtitle = []
	all_mean = []
	all_var = []
	all_mean_not = []
	all_var_not = []

	if compare_concat:
		np.save(os.path.join(path_ranking,'distances_concatenated.npy'), np.array(notSortedIndicesConcat))
		np.save(os.path.join(path_ranking,'rankings_relevant_concatenated.npy'),sortedIndicesConcat)
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesConcat,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		subtitle.append('Concatenated')
		all_mean.append(np.mean(mean_concat))
		all_var.append(np.mean(var_concat))
		all_mean_not.append(np.mean(non_mean_concat))
		all_var_not.append(np.mean(non_var_concat))

	if compare_ESS:
		np.save(os.path.join(path_ranking_ess,'distances_ess.npy'), np.array(notSortedIndicesESS)) 
		np.save(os.path.join(path_ranking_ess,'rankings_relevant_ess.npy'), np.array(sortedIndicesESS))
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesESS,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		subtitle.append('ESS')
		all_mean.append(np.mean(mean_ESS))
		all_var.append(np.mean(var_ESS))
		all_mean_not.append(np.mean(non_mean_ESS))
		all_var_not.append(np.mean(non_var_ESS))

	if compare_finetuned:
		np.save(os.path.join(path_ranking_finetuning,'distances_finetuned.npy'), np.array(notSortedIndicesFinetuning)) 
		np.save(os.path.join(path_ranking_finetuning,'rankings_relevant_finetuned.npy'),sortedIndicesFinetuning)
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesFinetuning,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		subtitle.append('Fine-Tuned')
		all_mean.append(np.mean(mean_fine))
		all_var.append(np.mean(var_fine))
		all_mean_not.append(np.mean(non_mean_fine))
		all_var_not.append(np.mean(non_var_fine))

	if compare_cross:
		np.save(os.path.join(path_ranking_crossEntropy,'distances_cross.npy'), np.array(notSortedIndicesCross)) 
		np.save(os.path.join(path_ranking_crossEntropy,'rankings_relevant_cross.npy'),sortedIndicesCross)
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesCross,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		subtitle.append('Cross-Entropy')
		all_mean.append(np.mean(mean_cross))
		all_var.append(np.mean(var_cross))
		all_mean_not.append(np.mean(non_mean_cross))
		all_var_not.append(np.mean(non_var_cross))

	if compare_contrastive:
		np.save(os.path.join(path_ranking_contrastive,'distances_contrastive.npy'), np.array(notSortedIndicesContrastive)) 
		np.save(os.path.join(path_ranking_contrastive,'rankings_relevant_contrastive.npy'),sortedIndicesContrastive)
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesContrastive,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		subtitle.append('Contrastive')
		all_mean.append(np.mean(mean_cont))
		all_var.append(np.mean(var_cont))
		all_mean_not.append(np.mean(non_mean_cont))
		all_var_not.append(np.mean(non_var_cont))

	if compare_triplet:
		np.save(os.path.join(path_ranking_triplet,'distances_triplet.npy'), np.array(notSortedIndicesTriplet)) 
		np.save(os.path.join(path_ranking_triplet,'rankings_relevant_triplet.npy'),sortedIndicesTriplet)
		precision_total, map_values, media, var, index_min,index_max = iterations_analysis(sortedIndicesTriplet,[],np.arange(0,relevant))
		media_final.append(np.copy(media))
		var_final.append(np.copy(var))
		subtitle.append('Triplet')
		all_mean.append(np.mean(mean_triplet))
		all_var.append(np.mean(var_triplet))
		all_mean_not.append(np.mean(non_mean_triplet))
		all_var_not.append(np.mean(non_var_triplet))

	## save mean distances and variance
	np.save(os.path.join(save_path_graph,'positive_mean_distances.npy'), np.array(all_mean))
	np.save(os.path.join(save_path_graph,'negative_mean_distances.npy'), np.array(all_mean_not))
	np.save(os.path.join(save_path_graph,'positive_var_distances.npy'), np.array(all_var))
	np.save(os.path.join(save_path_graph,'negative_var_distances.npy'), np.array(all_var_not))
	
	## save values for graph
	np.save(os.path.join(save_path_graph,'points_mean_graph.npy'), np.array(media_final))
	np.save(os.path.join(save_path_graph,'points_var_graph.npy'), np.array(var_final))

	## plot comparison
	cores = ['#62c9bf', '#ad7a99', '#746a9e','#d8b989', '#e47c7b','#91c4f2']
	mark_precision = ['o','<','*', '>', 'X', 'D']
	print(media_final)
	line_error_plot(media_final,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],var_final, len(subtitle), mark_precision,'Recall','Precision', subtitle,os.path.join(save_path_graph,'precision_recall_comparison.pdf'),cores,dataset)

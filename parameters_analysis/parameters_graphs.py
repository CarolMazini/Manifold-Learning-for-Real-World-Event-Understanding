#!/usr/bin/env python3
import numpy as np
from utils.set_statistics import *
from projections.projections import *
from ranking_utils.ranking_construction import *
from ranking_utils.rank_metrics import *




#the function evaluates the precision in each top k if points != []
#if points == [], the function finds the points for the porcentages of recall:
# [0.1, 0.2, 0.3, ..., 1.0]
def iterations_analysis(lista,points,relevantNum):
	np.set_printoptions(threshold=np.inf)
	print(len(lista))
	print(len(lista[0]))
	#calula ranking precision at points
	matrix = []
	map_aux = []

	precision_total = []
	binary_total = []


	for i in range(len(lista)):
		precision_aux = []

		try:
			if relevantNum > 1:
				binary, recallIndex = calculateIndexRecall(lista[i],relevantNum)

		except:

			binary, recallIndex =calculateIndexRecallbyVector(lista[i],relevantNum)

		if points == []:

			for j in range(10):
				precision_aux.append(precision_at_k(binary, int(recallIndex[j]+1)))
		else:
			for j in range(len(points)):


				precision_aux.append(precision_at_k(binary, points[j]))

		binary_total.append(np.copy(binary))
		precision_total.append(np.array(np.copy(precision_aux)))
		print(len(precision_total))
		print(len(precision_total[0]))


	map_values = []

	binary_total = np.array(binary_total)

	if points == []:
		map_values.append(mean_average_precision(binary_total))
	else:
		for j in range(len(points)):

			map_values.append(mean_average_precision(binary_total[:,0:points[j]]))


	precision_total = np.array(precision_total)

	media, var = statistics(precision_total)
	index_min, index_max = min_max(precision_total)

	media_matrix = []
	var_matrix = []

	media_matrix.append(media)
	var_matrix.append(var)
	map_final = []
	map_final.append(np.array(map_values))

	return precision_total, map_values, media, var, index_min,index_max







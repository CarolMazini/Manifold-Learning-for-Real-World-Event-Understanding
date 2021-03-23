#!/usr/bin/env python3
import numpy as np


def statistics(matrix):

	"""
	entrada: cada linha e um conjunto de resultados de uma iteracao
	saida: o vetor de media e variancia dos resultados
	"""

	media = np.mean(matrix, axis=0)
	var = np.var(matrix, axis = 0)

	return media, var

	
def min_max(matrix):

	"""
	entrada: matriz com um conjunto de resultados por linha
	saida: um conjunto de resultados com o maior numero de minimos e outro com o maior numero de maximos
	"""

	cont_max = np.zeros(len(matrix)) 
	cont_min = np.zeros(len(matrix)) 

	for i in range(len(matrix[0])):
		value_max = np.max(matrix[:,i])
		value_min = np.min(matrix[:,i])

		args_min = np.where(matrix[:,i]==value_min)
		args_max = np.where(matrix[:,i]==value_max)

		cont_min[args_min]+=1
		cont_max[args_max]+=1


	return np.argmax(cont_min),np.argmax(cont_max)


		

	
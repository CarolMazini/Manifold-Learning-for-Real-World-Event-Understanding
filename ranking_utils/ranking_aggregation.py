#!/usr/bin/env python3
import numpy as np
import math




##ranking aggregation methods
def simple_concat(lista,num):
	"""
	entrada: lista de rankings (posicoes ordenadas)
	saida: ranking final onde os rankings foram concatenados posicao a posicao (k1-ranking1,k1-ranking2,...)

	"""
	final_ranking = np.full(len(lista[0]),np.inf)
	index = 0
	for i in range(len(final_ranking)):
		for j in range(num):
			if not(lista[j][i] in final_ranking):
				final_ranking[index] = lista[j][i]
				index+=1

	return final_ranking


def rankingCombMin(matrix):
	"""
	entrada: matriz onde cada coluna e um ranking nao ordenado (distancias para uma das sementes)
	saida: vetor com a menor distancia obtida para a amostra dentre todas as sementes

	"""
	sumAll = np.zeros(len(matrix))

	for i in range(len(matrix)):
		sumAll[i] = np.min(matrix[i])

	return sumAll

def rankingCombSum(matrix):
	"""
	entrada: matriz onde cada coluna e um ranking nao ordenado (distancias para uma das sementes)
	saida: vetor com a soma das distancias obtidas para a amostra para todas as sementes

	"""
	sumAll = np.zeros(len(matrix))

	for i in range(len(matrix)):
		sumAll[i] = np.sum(matrix[i])

	return sumAll

def rankingCombMinSum(matrix,num_sum):
	"""
	entrada: matriz onde cada coluna e um ranking nao ordenado (distancias para uma das sementes)
	saida: vetor com a soma das distancias obtidas para a amostra para todas as sementes

	"""
	sumAll = np.zeros(len(matrix))
	matrix_aux = np.sort(matrix) 
	#print(matrix_aux)

	for i in range(len(matrix)):
		sumAll[i] = np.sum(matrix_aux[i,0:num_sum])

	return sumAll

def rankingMinPath(matrix,seeds,levels):

	"""
	errado, analisar
	"""

	chosen = np.full(matrix.shape[0], np.inf)
	height_vector = np.zeros(matrix.shape[0],dtype=int)
	parent = np.full(matrix.shape[0], np.inf)
	fila = []

	for s in seeds:
		fila.append(s)
		height_vector[s] = 0
		parent[s]=-1
		chosen[s] = 0
	print(fila)

	level_atual=0
	while len(fila)>0:
		index = fila[0] 
		if(height_vector[index]>level_atual):
			level_atual = level_atual+1

		for i in range(1, len(chosen)):
			if((matrix[index,i]+chosen[index])<chosen[i]):
				chosen[i] = (matrix[index,i]+chosen[index])
			if(height_vector[index]<levels):
				fila.append(i)
				height_vector[i] = height_vector[index]+1
		fila.pop(0)
		print(fila)
	return matrix

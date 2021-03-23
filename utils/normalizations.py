#!/usr/bin/env python3
import numpy as np
import math
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler


##scaling methods
def normZ(sumAll):
	"""
	entrada: matriz na qual as colunas precisam ter os atributos escalados
	saida: matriz de mesma dimensao de entrada com atributos das colunas escalados

	usa funcao do sklearn StandadScaller
	"""
   
	scaler = StandardScaler()
	sumAllNorm = scaler.fit_transform(sumAll)
	return(sumAllNorm)

def normQuartil(sumAll, num):

	"""
	entrada: matriz para ter as colunas escaladas e numero do percentil esperado (0-100)
	saida: matriz de mesma dimensao escalada

	divide cada elemento pelo percentil da coluna
	"""

	sumAll2 = np.zeros(sumAll.shape,float)
	for j in range(len(sumAll[0])):
		quartil = np.percentile(sumAll[:,j], num) 
		for i in range(len(sumAll)):
			sumAll2[i,j] = sumAll[i,j]/(quartil+0.00001)
			
	return sumAll2

def normMinMax(sumAll):

	"""
	entrada: matriz para escalar colunas
	saida: matriz de mesma dimensao com colunas escaladas

	encontra o valor maximo e minimo de cada coluna para escalar
	"""

	'''
	sumAll2 = np.zeros(sumAll.shape,float)
	for j in range(len(sumAll[0])):
		max_num = float(np.max(sumAll[:,j]))
		min_num = float(np.min(sumAll[:,j]))
		sumAll2[:,j] = (sumAll[:,j] - min_num)/(max_num - min_num+0.00001)

	'''
	scaler = MinMaxScaler()
	sumAllNorm = scaler.fit_transform(sumAll)
	return sumAllNorm


##normalize method
def normL2(sumAll):

	"""
	entrada: matriz para que as linhas sejam normalizadas
	saida: matriz de mesma dimensao com linhas normalizadas pela norma l2
	"""

	Normalizer(norm = 'l2').fit(sumAll)
	
	sumAllNorm = Normalizer(norm = 'l2').transform(sumAll)

	return(sumAllNorm)
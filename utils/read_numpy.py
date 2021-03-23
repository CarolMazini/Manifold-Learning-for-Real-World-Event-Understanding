#!/usr/bin/env python3
import numpy as np
import glob
import shutil
import os


def ready_directory(path_in):
	lista = []
	names = sorted(glob.glob(os.path.join(path_in, '*')))

	for n in names:
		element = np.load(n)
		index = n.split('_')
		index = int(index[-1].split('.')[0])
		lista.insert(index, element)

	return lista

def print_vector(path_in):
	vector = np.load(path_in)

	for i,value in enumerate(vector):
		print(str(i)+':',value)

	print('Tamanho:',len(vector))





		

	
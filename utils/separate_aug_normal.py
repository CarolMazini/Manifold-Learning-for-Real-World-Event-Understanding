import numpy as np
import math
import shutil
import os
import glob
import random
import os, sys, stat
from os import *


def sync_sets(general_names, general_features,delete):

	'''
	input: set of features and correspondent filenames, and names of files to be deleted
	output: lists of separated features and names in order without the expected to be deleted
	'''

	final_names = []
	final_features = []
	index = []

	aux = 0
	for tr in delete:
		index += [i for i, elem in enumerate(general_names) if tr in elem]

	for  i in range(len(general_names)):
		if i not in index:
			final_names.append(general_names[i])
			final_features.append(general_features[i,:])

	print(len(final_names))
	print(len(final_features))
	print((final_names[0]))
	print((final_features[0]))




	return np.array(final_names), np.array(final_features)






##################################MAIN#####################################

if __name__ == '__main__':

	classes = ['positive']
	dataset = ['bombing']
	split = ['train','val','test']
	weights = ['imagenet', 'places','reID']

	for c in classes:
		for d in dataset:
			for s in split:
				for w in weights:
					print(c,d,s,w)
					#dir_out = '/home/carolinerodrigues/featuresEvent/codes/out_codes/features_finetuning_cnn/'+d+'/'
					dir_out = '/home/carolinerodrigues/featuresEvent/'+d+'/features/split/'
					print(dir_out)
					#out_name = c+'_'+d+'_noAug'+s+'_names_'+w+'.npy'
					#out_features = c+'_'+d+'_noAug'+s+'_'+w+'.npy'
					out_name = c+s+'_names_'+w+'.npy'
					out_features = c+s+'_'+w+'.npy'
					names = np.load(dir_out+out_name)
					features = np.load(dir_out+out_features)

					delete = ['crop','rotated','zoom']

					final_names, final_features = sync_sets(names, features,delete)

					np.save(dir_out+'noAug_'+out_name,final_names)
					np.save(dir_out+'noAug_'+out_features,final_features)

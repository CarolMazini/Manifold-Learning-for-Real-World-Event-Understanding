import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random


def read_data(dframe, path):
	X = []

	try:
		for fn in tqdm(dframe.FileName.values):
			vec = np.load(os.path.join(path,fn+'.npy'))[0]
			X.append(vec[1:].astype(np.float64))
	except:
		try:
			for fn in tqdm(dframe.FileName.values):
				vec = np.load(os.path.join(path,fn+'.npy'))
				X.append(vec[1:].astype(np.float64))
		except:
			print('Impossible to read!!!')

	return np.array(X)

def get_positive_indexes(df):

	indexes_pos  = np.where(df.Label.values > 0)[0]

	#print(indexes_pos)

	return indexes_pos

def find_num_relevant(df):

	tam_relevant = len(get_positive_indexes(df))

	#print(tam_relevant)

	return tam_relevant

def get_random_seeds(df, num_seeds,random_seed):
	random.seed(a=random_seed)

	positive = get_positive_indexes(df)

	seeds = []

	for i in range(num_seeds):
		seed = random.choice(positive)
		while seed in seeds:
			seed = random.choice(positive)
		seeds.append(seed)

	#print(seeds)

	return seeds


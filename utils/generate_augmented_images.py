import numpy as np
import math
import glob
import random
import os, sys, stat
from save_inDir import * 
import transformations

#generate and save aumentation in separated folders
def generate_augmentation(filename, dir_save_fig):

	filename2 = sorted(glob.glob(os.path.join(filename, '*','*')))

	save_crop = '/crop/'
	if not os.path.isdir(dir_save_fig+save_crop):
		os.mkdir(dir_save_fig+save_crop)
	save_rotation = '/rotated/'
	if not os.path.isdir(dir_save_fig+save_rotation):
		os.mkdir(dir_save_fig+save_rotation)
	save_zoom = '/zoom/'
	if not os.path.isdir(dir_save_fig+save_zoom):
		os.mkdir(dir_save_fig+save_zoom)
	for i in range(5):
		random.seed(a=i)
		transformations.crop_generator(filename2, 224,dir_save_fig+save_crop)
		transformations.zoom_generator(filename2, 5.0,dir_save_fig+save_zoom)
		transformations.rotation_generator(filename2, 30,dir_save_fig+save_rotation)



##########################MAIN################################


filename = '../dataset/'
dataset = 'museu_nacional' #'wedding', 'fire', 'bombing' or 'bangladesh_fire'
n = 'train' #or val



name = os.path.join(filename, dataset,n)
df_paths = pd.read_csv(name+'.csv', header=0)


generate_augmentation(df_paths.path, name+'/')

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image
from multiprocessing import Process
import matplotlib.pyplot as plt
from  matplotlib.pyplot import cm
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from tqdm import tqdm
import time
import numpy as np
import scipy as sc
from sys import argv
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Input
from keras.models import Model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
import math
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.cluster import spectral_clustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pylab as pl
from sklearn.neighbors import kneighbors_graph
import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import shutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import h5py
import random
from PIL import Image
import PIL
#import cv2
import os
import glob
#import imutils




import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
if tf.get_default_session() is not None:
	status = tf.get_default_session().TF_NewStatus()
	tf.get_default_session().TF_DeleteDeprecatedSession(self._session, status)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True	#avoid getting all available memory in GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.5  #uncomment to limit GPU allocation
config.gpu_options.visible_device_list = "1"  #set which GPU to use
set_session(tf.Session(config=config))
HDF5_USE_FILE_LOCKING='FALSE'



#given two sets of filenames, order the second according to the first (split in 3 sets)
def sync_sets(general_names, general_features,train,val,test):

	'''
	input: set of features and correspondent filenames, and names of files in each of the three sets: train, validation and test
	output: lists of separated features and names in order
	'''

	train_names = []
	train_features = []
	val_names = []
	val_features = []
	test_names = []
	test_features = []

	aux = 0
	for tr in train:
		index = [i for i, elem in enumerate(general_names) if tr in elem[0]]
		if(index != []):
			print(tr)
			train_names.append(general_names[index[0],0])
			train_features.append(general_features[index[0],:])
		else:

			print('error')

	for v in val:
		index = [i for i, elem in enumerate(general_names) if v in elem[0]]
		if(index != []):
			print(v)
			val_names.append(general_names[index[0],0])
			val_features.append(general_features[index[0],:])
		else:

			print('error')

	for t in test:
		index = [i for i, elem in enumerate(general_names) if t in elem[0]]
		if(index != []):
			print(t)
			test_names.append(general_names[index[0],0])
			test_features.append(general_features[index[0],:])
		else:
			print('error')


	return np.array(train_names), np.array(train_features), np.array(val_names), np.array(val_features), np.array(test_names), np.array(test_features)



def split_sets(general_names, general_features,val,test):

	train_names = []
	train_features = []
	val_names = []
	val_features = []
	test_names = []
	test_features = []

	for name_index in range(len(general_names)):
		#print(general_names[name_index])
		aux = 0
		for v in val:
			#print(v)
			if v in general_names[name_index]:
				#print('entrou v')
				val_names.append(general_names[name_index])
				val_features.append(general_features[name_index])
				aux = 1
				break
		if aux == 0:
			for t in test:
				#print(t)
				if t in general_names[name_index]:
					#print('entrou t')
					test_names.append(general_names[name_index])
					test_features.append(general_features[name_index])
					aux = 1
					break
		if aux == 0:
			#print('entrou train')
			train_names.append(general_names[name_index])
			train_features.append(general_features[name_index])


	return np.array(train_names), np.array(train_features), np.array(val_names), np.array(val_features), np.array(test_names), np.array(test_features)

def get_features_separated(generatorImagenet,generatorPlaces,feature_extractor_imagenet,feature_extractor_places,val, test,name):

	general_names = generatorImagenet.filenames

	np.save(name+"_filenames.npy", generatorImagenet.filenames)

	try:
		features_places = np.load(name + "train_places.npy")
		print('complete features', len(features_places))
		print("feature loading finished")
	except:
		features_places = feature_extractor_places.predict_generator(generatorPlaces, steps=generatorPlaces.samples/generatorPlaces.batch_size,verbose=1)
		print('complete features', len(features_places))
		train_names, train_features, val_names, val_features, test_names, test_features = split_sets(general_names, features_places,val,test)
		print('train names: ',len(train_names))
		print('train features: ',len(train_features))
		print('val names: ',len(val_names))
		print('val features: ',len(val_features))
		print('test names: ',len(test_names))
		print('test features: ',len(test_features))
		print(" feature generator finished")
		np.save(name+"train_places.npy", train_features)
		np.save(name+"val_places.npy", val_features)
		np.save(name+"test_places.npy", test_features)
		np.save(name+"train_names_places.npy", train_names)
		np.save(name+"val_names_places.npy", val_names)
		np.save(name+"test_names_places.npy", test_names)
		del(features_places)

	try:
		features_imagenet = np.load(name + "train_imagenet.npy")
		print("feature loading finished")
	except:
		features_imagenet = feature_extractor_imagenet.predict_generator(generatorImagenet, steps=generatorImagenet.samples/generatorImagenet.batch_size,verbose=1)
		print('complete features', len(features_imagenet))
		train_names, train_features, val_names, val_features, test_names, test_features = split_sets(general_names, features_imagenet,val,test)
		print('train names: ',len(train_names))
		print('train features: ',len(train_features))
		print('val names: ',len(val_names))
		print('val features: ',len(val_features))
		print('test names: ',len(test_names))
		print('test features: ',len(test_features))
		print(" feature generator finished")
		np.save(name+"train_imagenet.npy", train_features)
		np.save(name+"val_imagenet.npy", val_features)
		np.save(name+"test_imagenet.npy", test_features)
		np.save(name+"train_names_imagenet.npy", train_names)
		np.save(name+"val_names_imagenet.npy", val_names)
		np.save(name+"test_names_imagenet.npy", test_names)
		print(" feature generator finished")
		del(features_imagenet)

if __name__ == '__main__':

    filename_train = sorted(glob.glob(os.path.join('/home/carolinerodrigues/royalWedding/confirmation/positive/train', '*','*')))

    print(len(filename_train))
    print(filename_train[0])

    filename_val = sorted(glob.glob(os.path.join('/home/carolinerodrigues/royalWedding/confirmation/positive/val', '*','*')))

    print(len(filename_val))
    print(filename_val[0])

    filename_test = sorted(glob.glob(os.path.join('/home/carolinerodrigues/royalWedding/confirmation/positive/test', '*','*')))

    print(len(filename_test))
    print(filename_test[0])

    name_train = np.load('/home/carolinerodrigues/featuresEvent/wedding/features/split/noAug_positivetrain_names_places.npy')
    name_val = np.load('/home/carolinerodrigues/featuresEvent/wedding/features/split/noAug_positiveval_names_places.npy')
    name_test = np.load('/home/carolinerodrigues/featuresEvent/wedding/features/split/noAug_positivetest_names_places.npy')


    # for i in range(len(name_train)):
    #     name_train[i] = name_train[i].split('/')[-2]+'/'+name_train[i].split('/')[-1]
    
    # for i in range(len(name_val)):
    #     name_val[i] = name_val[i].split('/')[-2]+'/'+name_val[i].split('/')[-1]

    # for i in range(len(name_test)):
    #     name_test[i] = name_test[i].split('/')[-2]+'/'+name_test[i].split('/')[-1]

    print(len(name_train))
    print(name_train[0])

    print(len(name_val))
    print(name_val[0])

    print(len(name_test))
    print(name_test[0])

    erro_sem = 0
    erro_com = 0
    acerto = 0
    
    for filename in filename_train:
        flag = 0
        for train in name_train:
            if(train in filename):
                acerto+=1
                flag = 1
                break
        if flag == 0:
            erro_sem+=1

        for val in name_val:
            if(val in filename):
                erro_com+=1
                break

        for test in name_test:
            if(test in filename):
                erro_com+=1
                break

    print('Erro com train:', erro_com)
    print('Erro sem train:', erro_sem)
    print('Acerto train:',acerto)

    erro_sem = 0
    erro_com = 0
    acerto = 0
    
    for filename in filename_val:
        flag = 0
        for val in name_val:
            if(val in filename):
                acerto+=1
                flag = 1
                break
        if flag == 0:
            erro_sem+=1

        for train in name_train:
            if(train in filename):
                erro_com+=1
                break

        for test in name_test:
            if(test in filename):
                erro_com+=1
                break

    print('Erro com val:', erro_com)
    print('Erro sem val:', erro_sem)
    print('Acerto val:',acerto)

    erro_sem = 0
    erro_com = 0
    acerto = 0
    
    for filename in filename_test:
        flag = 0
        for test in name_test:
            if(test in filename):
                acerto+=1
                flag = 1
                break
        if flag == 0:
            erro_sem+=1

        for val in name_val:
            if(val in filename):
                erro_com+=1
                break

        for train in name_train:
            if(train in filename):
                erro_com+=1
                break

    print('Erro com test:', erro_com)
    print('Erro sem test:', erro_sem)
    print('Acerto test:',acerto)

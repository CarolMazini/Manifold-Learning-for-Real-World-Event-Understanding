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
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
HDF5_USE_FILE_LOCKING='FALSE'

def set_gpu(gpu_id):
	if tf.get_default_session() is not None:
		status = tf.get_default_session().TF_NewStatus()
		tf.get_default_session().TF_DeleteDeprecatedSession(self._session, status)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	#avoid getting all available memory in GPU
	config.gpu_options.per_process_gpu_memory_fraction = 0.5  #uncomment to limit GPU allocation
	config.gpu_options.visible_device_list = gpu_id  #set which GPU to use
	set_session(tf.Session(config=config))





def loadNetworkPlaces(train_layer, weights, finetuned = False):
	from keras.applications.vgg16 import VGG16, preprocess_input

	print("Loading VGG16 weights from Places365...")
	feature_extractor = VGG16(include_top=True, weights='vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5', input_tensor=None, input_shape=None, pooling=None, classes=365)
	#print(feature_extractor.summary())
	# Freeze the layers except the last train_layers
	for layer in feature_extractor.layers[:-train_layer]:
		layer.trainable = False

	for layer in feature_extractor.layers:
		print(layer, layer.trainable)


	feature_extractor.layers.pop()
	feature_extractor.outputs = [feature_extractor.layers[-1].output]
	feature_extractor.layers[-1].outbound_nodes = []

	#print(feature_extractor.summary())
	if finetuned:
		x = feature_extractor.layers[-1].output

		x = Dense(2, activation='softmax')(x)
		feature_extractor = Model(inputs=feature_extractor.inputs, outputs=x)

		feature_extractor.load_weights(weights)

		#print(feature_extractor.summary())

		feature_extractor.layers.pop()
		feature_extractor.outputs = [feature_extractor.layers[-1].output]
		feature_extractor.layers[-1].outbound_nodes = []

	print(feature_extractor.summary())

	print("Loading VGG Places done.")

	return feature_extractor

def loadNetworkImagenet(train_layer, weights,finetuned = False):
	from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

	print("Loading InceptionResnet weights from Imagenet...")

	feature_extractor = InceptionResNetV2(include_top=True, weights='inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	#print(feature_extractor.summary())
	# Freeze the layers except the last train_layers
	for layer in feature_extractor.layers[:-train_layer]:
		layer.trainable = False

	for layer in feature_extractor.layers:
            print(layer, layer.trainable)

	feature_extractor.layers.pop()
	feature_extractor.outputs = [feature_extractor.layers[-1].output]
	feature_extractor.layers[-1].outbound_nodes = []

	#print(feature_extractor.summary())
	if finetuned:
		x = feature_extractor.layers[-1].output

		x = Dense(2, activation='softmax')(x)
		feature_extractor = Model(inputs=feature_extractor.inputs, outputs=x)

		feature_extractor.load_weights(weights)

		feature_extractor.layers.pop()
		feature_extractor.outputs = [feature_extractor.layers[-1].output]
		feature_extractor.layers[-1].outbound_nodes = []

	print(feature_extractor.summary())

	print("Loading InceptionResnet imagenet done.")

	return feature_extractor


#generate data from training and validation without augmentation
def generateData(size,dir):

	#size VGG 224
	#size Imagenet 299

	print("Generating features...")

	datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input
	)

	generator = datagen.flow_from_directory(
		dir,
		seed = 42,
		target_size=(size, size),
		batch_size=16,
		class_mode='categorical',
		shuffle = False)

	return generator
	
	
def generateData_fromFile(size,dataframe):

	#size VGG 224
	#size Imagenet 299

	print("Generating features...")

	datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input
	)

	generator = datagen.flow_from_dataframe(
				dataframe, directory=None, x_col='path', y_col=None,
    				target_size=(size, size),
    				#label_mode="int",
				class_mode=None,
    				batch_size=16,
    				shuffle=False,
    				validate_filenames=False)

	return generator


def getFeatures(generatorImagenet,generatorPlaces,feature_extractor_imagenet,feature_extractor_places):


	features_places = feature_extractor_places.predict_generator(generatorPlaces, steps=generatorPlaces.samples/generatorPlaces.batch_size,verbose=1)
		
	features_imagenet = feature_extractor_imagenet.predict_generator(generatorImagenet, steps=generatorImagenet.samples/generatorImagenet.batch_size,verbose=1)
	return features_imagenet,features_places




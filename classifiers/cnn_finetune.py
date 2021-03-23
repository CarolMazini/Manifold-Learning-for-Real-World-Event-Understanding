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
import os
import keras
from keras.callbacks import ModelCheckpoint


HDF5_USE_FILE_LOCKING='FALSE'

def set_gpu(gpu_id):
        if tf.get_default_session() is not None:
                status = tf.get_default_session().TF_NewStatus()
                tf.get_default_session().TF_DeleteDeprecatedSession(self._session, status)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  #avoid getting all available memory in GPU
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  #uncomment to limit GPU allocation
        config.gpu_options.visible_device_list = gpu_id  #set which GPU to use
        set_session(tf.Session(config=config))



def loadNetwork(tipo_features, weights):

	print("Loading weights...")
	if tipo_features == 'objects':
		from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
		model = InceptionResNetV2(include_top=True, weights=weights, input_tensor=None, input_shape=None, pooling=None, classes=1000)
		train_layer = 1
	elif tipo_features == 'places':
		from keras.applications.vgg16 import VGG16, preprocess_input
		model = VGG16(include_top=True, weights=weights, input_tensor=None, input_shape=None, pooling=None, classes=365)
		train_layer = 3
	

	# Freeze the layers except the last train_layers
	for layer in model.layers[:-train_layer]:
		layer.trainable = False

	for layer in model.layers:
		print(layer, layer.trainable)

	model.layers.pop()
	model.outputs = [model.layers[-1].output]
	model.layers[-1].outbound_nodes = []

	x = model.layers[-1].output

	#comentar linha seguinte pra usar o modelo convernsional
	#x = Dense(512, activation='relu')(x)
	x = Dense(2, activation='softmax')(x)
	model = Model(inputs=model.inputs, outputs=x)

	print("Loading done.")

	return model



#generate data from training and validation without augmentation
def generateData_fromFile(size,df,tipo_features):

	#size VGG 224
	#size Imagenet 299
	if tipo_features == 'objects':
		from keras.applications.inception_resnet_v2 import preprocess_input
	elif tipo_features == 'places':
		from keras.applications.vgg16 import preprocess_input

	print("Generating features...")

	datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input
	)

	generator = datagen.flow_from_dataframe(
				dataframe= df, directory=None, x_col='path', y_col="label",seed = 42,
				target_size=(size, size),
				#label_mode="int",
				class_mode='categorical',
				batch_size=16,
				shuffle=True,
				validate_filenames=None)

	return generator



def train_model(model,data,data_val,epochs,name_path):

	# Specify the training configuration (optimizer, loss, metrics)
	# model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
	# 			# Loss function to minimize
	# 			loss=keras.losses.SparseCategoricalCrossentropy(),
	# 			# List of metrics to monitor
	# 			metrics=[keras.metrics.SparseCategoricalAccuracy()])


	adam = keras.optimizers.Adam(lr=0.00001, decay=1e-5)

	model.compile(optimizer=adam,
				  loss='categorical_crossentropy', metrics=['accuracy'])

	# Train the model by slicing the data into "batches"
	# of size "batch_size", and repeatedly iterating over
	# the entire dataset for a given number of "epochs"
	print('# Fit model on training data')




	history = model.fit_generator(
		data,
		steps_per_epoch=data.samples/data.batch_size,
		epochs=epochs,
		validation_data=data_val,
		validation_steps=data_val.samples/data_val.batch_size,
		callbacks=[ModelCheckpoint(filepath=name_path, save_best_only=True,save_weights_only=True)])

	# The returned "history" object holds a record
	# of the loss values and metric values during training
	print('\nhistory dict:', history.history)
	return model



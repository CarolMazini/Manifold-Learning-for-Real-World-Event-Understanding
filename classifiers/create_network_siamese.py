from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.layers import Input,concatenate
from keras.callbacks import ModelCheckpoint
import copy


num_classes = 2
epochs = 50


def euclidean_distance(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, indices):
	'''Positive and negative pair creation.
	Alternates between positive and negative pairs.
	'''
	data1 = []
	data2 = []
	data3 = []
	data4 = []
	data5 = []
	data6 = []
	labels = []
	n = min([len(indices[d]) for d in range(num_classes)]) - 1
	for j in range(5):
		for d in range(num_classes):
			for i in range(n):
				z1, z2 = indices[d][i], indices[d][i + 1]
				data1.append(x[0][z1])
				data2.append(x[1][z1])
				data3.append(x[2][z1])
				data4.append(x[0][z2])
				data5.append(x[1][z2])
				data6.append(x[2][z2])
				labels.append(1)
				inc = random.randrange(1, num_classes)
				dn = (d + inc) % num_classes
				z1, z2 = indices[d][i], indices[dn][i]
				data1.append(x[0][z1])
				data2.append(x[1][z1])
				data3.append(x[2][z1])
				data4.append(x[0][z2])
				data5.append(x[1][z2])
				data6.append(x[2][z2])
				labels.append(0)
	return np.array(data1),np.array(data2),np.array(data3),np.array(data4),np.array(data5),np.array(data6), np.array(labels)


def create_pairs2(x, num_positive):
	'''Positive and negative pair creation.
	Alternates between positive and negative pairs.
	'''
	data1 = []
	data2 = []
	data3 = []
	data4 = []
	data5 = []
	data6 = []
	labels = []

	for j in range(10):
		for i in range(num_positive):
			vec = np.arange(0, num_positive)
			vec[i] = i+1
			inc = random.choice(vec)
			z1, z2 = i, i+j
			data1.append(x[0][z1])
			data2.append(x[1][z1])
			data3.append(x[2][z1])
			data4.append(x[0][z2])
			data5.append(x[1][z2])
			data6.append(x[2][z2])
			labels.append(1)
			inc = random.randrange(num_positive, len(x[0]))
			z1, z2 = i, inc
			data1.append(x[0][z1])
			data2.append(x[1][z1])
			data3.append(x[2][z1])
			data4.append(x[0][z2])
			data5.append(x[1][z2])
			data6.append(x[2][z2])
			labels.append(0)
	return np.array(data1),np.array(data2),np.array(data3),np.array(data4),np.array(data5),np.array(data6), np.array(labels)


def create_base_network(input_dim):
	'''Base network to be shared (eq. to feature extraction).
	'''

	inputs1 = Input(shape=(input_dim[0],), name='places')
	inputs2 = Input(shape=(input_dim[1],), name='objects')
	inputs3 = Input(shape=(input_dim[2],), name='people')

	output_pre_middle_1 = Dense(512, activation='relu', name='pre_middle_places')(inputs1)
	output_pre_middle_2 = Dense(512, activation='relu', name='pre_middle_objects')(inputs2)
	output_pre_middle_3 = Dense(512, activation='relu', name='pre_middle_people')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])


	output_middle = Dense(128, activation='relu',name = 'middle')(x)

	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=output_middle)


	print(model.summary())

	return model

#512/128/64
def create_base_network_deep1(input_dim):
	'''Base network to be shared (eq. to feature extraction).
	'''

	inputs1 = Input(shape=(input_dim[0],), name='places')
	inputs2 = Input(shape=(input_dim[1],), name='objects')
	inputs3 = Input(shape=(input_dim[2],), name='people')

	output_pre_middle_1 = Dense(512, activation='relu', name='pre_middle_places')(inputs1)
	output_pre_middle_2 = Dense(512, activation='relu', name='pre_middle_objects')(inputs2)
	output_pre_middle_3 = Dense(512, activation='relu', name='pre_middle_people')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])


	output_middle = Dense(128, activation='relu',name = 'middle_pre')(x)

	output_middle_pos = Dense(64, activation='relu',name = 'middle')(output_middle)

	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=output_middle_pos)


	print(model.summary())

	return model


#1024/512
def create_base_network_deep2(input_dim):
	'''Base network to be shared (eq. to feature extraction).
	'''

	inputs1 = Input(shape=(input_dim[0],), name='places')
	inputs2 = Input(shape=(input_dim[1],), name='objects')
	inputs3 = Input(shape=(input_dim[2],), name='people')

	output_pre_middle_1 = Dense(1024, activation='relu', name='pre_middle_places')(inputs1)
	output_pre_middle_2 = Dense(1024, activation='relu', name='pre_middle_objects')(inputs2)
	output_pre_middle_3 = Dense(1024, activation='relu', name='pre_middle_people')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])


	output_middle = Dense(512, activation='relu',name = 'middle')(x)

	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=output_middle)


	print(model.summary())

	return model


#1024/512/128
def create_base_network_deep3(input_dim):
	'''Base network to be shared (eq. to feature extraction).
	'''

	inputs1 = Input(shape=(input_dim[0],), name='places')
	inputs2 = Input(shape=(input_dim[1],), name='objects')
	inputs3 = Input(shape=(input_dim[2],), name='people')

	output_pre_middle_1 = Dense(1024, activation='relu', name='pre_middle_places')(inputs1)
	output_pre_middle_2 = Dense(1024, activation='relu', name='pre_middle_objects')(inputs2)
	output_pre_middle_3 = Dense(1024, activation='relu', name='pre_middle_people')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])


	output_middle = Dense(512, activation='relu',name = 'middle_pre')(x)

	output_middle_pos = Dense(128, activation='relu',name = 'middle')(output_middle)

	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=output_middle_pos)


	print(model.summary())

	return model



def extract_model(model,layer_name):
	print(model.layers[-1].name)
	#print((model.layers[-2]).get_layer(layer_name).name)

	model.layers.pop()
	model.outputs = [model.layers[-1].get_output_at(1)]
	model.layers[-1].outbound_nodes = []



	return model


def compute_accuracy(y_true, y_pred):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	pred = y_pred.ravel() < 0.5
	return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def prepare_data(train_positive_data, train_negative_data):

	train_places = np.concatenate([train_positive_data[0], train_negative_data[0]], axis=0)
	train_imagenet = np.concatenate([train_positive_data[1], train_negative_data[1]], axis=0)
	train_reID = np.concatenate([train_positive_data[2], train_negative_data[2]], axis=0)

	y_train = np.zeros(len(train_places))
	y_train[0:len(train_positive_data[0])] = 1

	# create training+test positive and negative pairs
	indices = [np.where(y_train == i)[0] for i in range(num_classes)]
	num_positive = len(train_positive_data[0])
	tr1,tr2,tr3,tr4,tr5,tr6, tr_y = create_pairs2([train_places,train_imagenet,train_reID], num_positive)

	return tr1,tr2,tr3,tr4,tr5,tr6, tr_y

def train_siamese(train_positive_data, train_negative_data, val_positive_data, val_negative_data, input_shape,name_path,type_net):

	tr1,tr2,tr3,tr4,tr5,tr6, tr_y = prepare_data(train_positive_data, train_negative_data)
	te1,te2,te3,te4,te5,te6, te_y = prepare_data(val_positive_data, val_negative_data)

	if type_net == 0:
		# network definition
		base_network = create_base_network(input_shape)
	elif type_net == 1:
		# network definition
		base_network = create_base_network_deep1(input_shape)
	elif type_net == 2:
		# network definition
		base_network = create_base_network_deep2(input_shape)
	else:
		# network definition
		base_network = create_base_network_deep3(input_shape)

	
	inputs1_a = Input(shape=(input_shape[0],), name='places_a')
	inputs2_a = Input(shape=(input_shape[1],), name='objects_a')
	inputs3_a = Input(shape=(input_shape[2],), name='people_a')

	inputs1_b = Input(shape=(input_shape[0],), name='places_b')
	inputs2_b = Input(shape=(input_shape[1],), name='objects_b')
	inputs3_b = Input(shape=(input_shape[2],), name='people_b')

	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network([inputs1_a, inputs2_a,inputs3_a])
	processed_b = base_network([inputs1_b, inputs2_b, inputs3_b])

	distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	model = Model([inputs1_a, inputs2_a,inputs3_a, inputs1_b, inputs2_b, inputs3_b], distance)
	print(model.summary())

	# train
	#rms = RMSprop()
	adam = Adam(lr=0.00001, decay=1e-5)
	model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])
	model.fit([tr1,tr2,tr3,tr4,tr5,tr6], tr_y,
			  batch_size=128,
			  epochs=epochs,
			  validation_data=([te1,te2,te3,te4,te5,te6], te_y),
			  callbacks=[ModelCheckpoint(filepath=name_path, save_best_only=True,save_weights_only=True)])

	# compute final accuracy on training and test sets
	y_pred = model.predict([tr1,tr2,tr3,tr4,tr5,tr6])
	tr_acc = compute_accuracy(tr_y, y_pred)
	y_pred = model.predict([te1,te2,te3,te4,te5,te6])
	te_acc = compute_accuracy(te_y, y_pred)

	print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
	print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

	return model

def extract_features_siamese(test_positive_data, test_negative_data, layer_name,input_shape,name_path, type_net):

	if type_net == 0:
		# network definition
		base_network = create_base_network(input_shape)
	elif type_net == 1:
		# network definition
		base_network = create_base_network_deep1(input_shape)
	elif type_net == 2:
		# network definition
		base_network = create_base_network_deep2(input_shape)
	else:
		# network definition
		base_network = create_base_network_deep3(input_shape)


	inputs1_a = Input(shape=(input_shape[0],), name='places_a')
	inputs2_a = Input(shape=(input_shape[1],), name='objects_a')
	inputs3_a = Input(shape=(input_shape[2],), name='people_a')

	inputs1_b = Input(shape=(input_shape[0],), name='places_b')
	inputs2_b = Input(shape=(input_shape[1],), name='objects_b')
	inputs3_b = Input(shape=(input_shape[2],), name='people_b')

	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network([inputs1_a, inputs2_a,inputs3_a])
	processed_b = base_network([inputs1_b, inputs2_b, inputs3_b])

	distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	model = Model([inputs1_a, inputs2_a,inputs3_a, inputs1_b, inputs2_b, inputs3_b], distance)

	print(model.summary())

	if name_path!='':
		print('Carregou!')
		model.load_weights(name_path)

	intermediate_model = extract_model(model,layer_name)

	features_positive = intermediate_model.predict([test_positive_data[0],test_positive_data[1],test_positive_data[2],test_positive_data[0],test_positive_data[1],test_positive_data[2]])
	features_negative = intermediate_model.predict([test_negative_data[0],test_negative_data[1],test_negative_data[2],test_negative_data[0],test_negative_data[1],test_negative_data[2]])

	return features_positive, features_negative

def predict_distances_siamese(test_data, layer_name,input_shape,name_path, seed):


	base_network = create_base_network(input_shape)

	inputs1_a = Input(shape=(input_shape[0],), name='places_a')
	inputs2_a = Input(shape=(input_shape[1],), name='objects_a')
	inputs3_a = Input(shape=(input_shape[2],), name='people_a')

	inputs1_b = Input(shape=(input_shape[0],), name='places_b')
	inputs2_b = Input(shape=(input_shape[1],), name='objects_b')
	inputs3_b = Input(shape=(input_shape[2],), name='people_b')

	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network([inputs1_a, inputs2_a,inputs3_a])
	processed_b = base_network([inputs1_b, inputs2_b, inputs3_b])

	distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	model = Model([inputs1_a, inputs2_a,inputs3_a, inputs1_b, inputs2_b, inputs3_b], distance)

	print(model.summary())

	model.load_weights(name_path)

	seed_data0 = []
	seed_data1 = []
	seed_data2 = []

	for i in range(len(test_data[0])):
		seed_data0.append(test_data[0][seed])
		seed_data1.append(test_data[1][seed])
		seed_data2.append(test_data[2][seed])
	seed_data0 = np.array(seed_data0)
	seed_data1 = np.array(seed_data1)
	seed_data2 = np.array(seed_data2)

	distances = model.predict([test_data[0],test_data[1],test_data[2],seed_data0,seed_data1,seed_data2])

	return distances

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.layers import Input,concatenate, Layer
from keras.callbacks import ModelCheckpoint
import copy
import time
from keras import regularizers
from sklearn.metrics import roc_auc_score, roc_curve

num_classes = 2
epochs = 5000
evaluate_every = 1
nb_classes = 10


def euclidean_distance(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

def compute_dist(a,b):
	return np.sum(np.square(a-b))

def get_batch_random(positive_train, negative_train, batch_size,s="train"):
	"""
	Create batch of APN triplets with a complete random strategy

	Arguments:
	batch_size -- integer
	Returns:
	triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
	"""

	X_positive = positive_train
	X_negative = negative_train


	# initialize result
	triplets_A_pl=[]
	triplets_A_ob=[]
	triplets_A_pe=[]
	triplets_P_pl=[]
	triplets_P_ob=[]
	triplets_P_pe=[]
	triplets_N_pl=[]
	triplets_N_ob=[]
	triplets_N_pe=[]

	#for i in range(batch_size//2):
	for i in range(batch_size):

		##positive as anchor
		#Pick two different random pics for this class => A and P
		[idx_A,idx_P] = np.random.choice(len(X_positive[0]),size=2,replace=False)

		#Pick a random pic for this negative class => N
		idx_N = np.random.randint(0, len(X_negative[0]))

		triplets_A_pl.append(X_positive[0][idx_A])
		triplets_A_ob.append(X_positive[1][idx_A])
		triplets_A_pe.append(X_positive[2][idx_A])
		triplets_P_pl.append(X_positive[0][idx_P])
		triplets_P_ob.append(X_positive[1][idx_P])
		triplets_P_pe.append(X_positive[2][idx_P])
		triplets_N_pl.append(X_negative[0][idx_N])
		triplets_N_ob.append(X_negative[1][idx_N])
		triplets_N_pe.append(X_negative[2][idx_N])


		

	return [np.array(triplets_A_pl),np.array(triplets_A_ob),np.array(triplets_A_pe)],[np.array(triplets_P_pl),np.array(triplets_P_ob),np.array(triplets_P_pe)],[np.array(triplets_N_pl),np.array(triplets_N_ob),np.array(triplets_N_pe)]

def drawTriplets(tripletbatch, nbmax=None):
	"""display the three images for each triplets in the batch
	"""
	labels = ["Anchor", "Positive", "Negative"]

	if (nbmax==None):
		nbrows = tripletbatch[0].shape[0]
	else:
		nbrows = min(nbmax,tripletbatch[0].shape[0])

	for row in range(nbrows):
		fig=plt.figure(figsize=(16,2))

		for i in range(3):
			subplot = fig.add_subplot(1,3,i+1)
			axis("off")
			plt.imshow(tripletbatch[i][row,:,:,0],vmin=0, vmax=1,cmap='Greys')
			subplot.title.set_text(labels[i])

def get_batch_hard(positive_train, negative_train,draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="train"):
	"""
	Create batch of APN "hard" triplets

	Arguments:
	draw_batch_size -- integer : number of initial randomly taken samples
	hard_batchs_size -- interger : select the number of hardest samples to keep
	norm_batchs_size -- interger : number of random samples to add
	Returns:
	triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
	"""
	X_positive = positive_train
	X_negative = negative_train


	# initialize result
	triplets=[]


	#Step 1 : pick a random batch to study
	studybatch_A,studybatch_P,studybatch_N = get_batch_random(positive_train, negative_train,draw_batch_size,s)

	print(studybatch_A)
	print(len(studybatch_A))

	#Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
	studybatchloss = np.zeros((draw_batch_size))

	#Compute embeddings for anchors, positive and negatives
	A = network.predict(studybatch_A)
	P = network.predict(studybatch_P)
	N = network.predict(studybatch_N)

	#Compute d(A,P)-d(A,N)
	studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)

	#Sort by distance (high distance first) and take the
	selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]

	#Draw other random samples from the batch
	selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)

	selection = np.append(selection,selection2)

	triplets = [studybatch_A[0][selection,:],studybatch_A[1][selection,:],studybatch_A[2][selection,:], studybatch_P[0][selection,:],studybatch_P[1][selection,:],studybatch_P[2][selection,:], studybatch_N[0][selection,:],studybatch_N[1][selection,:],studybatch_N[2][selection,:]]

	return triplets


def extract_model(model,layer_name):
	print(model.layers[-1].name)
	
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



class TripletLossLayer(Layer):
	def __init__(self, alpha, **kwargs):
		self.alpha = alpha
		super(TripletLossLayer, self).__init__(**kwargs)

	def triplet_loss(self, inputs):
		anchor, positive, negative = inputs
		p_dist = K.sum(K.square(anchor-positive), axis=-1)
		n_dist = K.sum(K.square(anchor-negative), axis=-1)
		return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

	def call(self, inputs):
		loss = self.triplet_loss(inputs)
		self.add_loss(loss)
		return loss


def build_network(input_shape):
	'''
	Define the neural network to learn image similarity
	Input :
			input_shape : shape of input images
			embeddingsize : vectorsize used to encode our picture
	'''
	inputs1 = Input(shape=(input_shape[0],), name='places')
	inputs2 = Input(shape=(input_shape[1],), name='objects')
	inputs3 = Input(shape=(input_shape[2],), name='people')

	
	output_pre_middle_1 = Dense(512, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_places')(inputs1)
	output_pre_middle_2 = Dense(512, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_objects')(inputs2)
	output_pre_middle_3 = Dense(512, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_people')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])
	

	output_middle = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(1e-3),kernel_initializer='he_uniform',name = 'middle')(x)

	distance = Lambda(lambda x: K.l2_normalize(x,axis=-1))(output_middle)

	
	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=distance)

	print(model.summary())

	return model

#512/128/64
def build_network_deep1(input_shape):
	'''
	Define the neural network to learn image similarity
	Input :
			input_shape : shape of input images
			embeddingsize : vectorsize used to encode our picture
	'''
	inputs1 = Input(shape=(input_shape[0],), name='places')
	inputs2 = Input(shape=(input_shape[1],), name='objects')
	inputs3 = Input(shape=(input_shape[2],), name='people')


	output_pre_middle_1 = Dense(512, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_places')(inputs1)
	output_pre_middle_2 = Dense(512, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_objects')(inputs2)
	output_pre_middle_3 = Dense(512, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_people')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])
	
	output_middle = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(1e-3),kernel_initializer='he_uniform',name = 'middle_pre')(x)

	output_middle_pos = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(1e-3),kernel_initializer='he_uniform',name = 'middle')(output_middle)

	distance = Lambda(lambda x: K.l2_normalize(x,axis=-1))(output_middle_pos)

	
	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=distance)

	print(model.summary())

	return model


#1024/512
def build_network_deep2(input_shape):
	'''
	Define the neural network to learn image similarity
	Input :
			input_shape : shape of input images
			embeddingsize : vectorsize used to encode our picture
	'''
	inputs1 = Input(shape=(input_shape[0],), name='places')
	inputs2 = Input(shape=(input_shape[1],), name='objects')
	inputs3 = Input(shape=(input_shape[2],), name='people')


	output_pre_middle_1 = Dense(1024, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_places')(inputs1)
	output_pre_middle_2 = Dense(1024, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_objects')(inputs2)
	output_pre_middle_3 = Dense(1024, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_people')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])
	
	output_middle = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(1e-3),kernel_initializer='he_uniform',name = 'middle')(x)

	
	distance = Lambda(lambda x: K.l2_normalize(x,axis=-1))(output_middle)

	
	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=distance)

	print(model.summary())

	return model


#1024/512/128
def build_network_deep3(input_shape):
	'''
	Define the neural network to learn image similarity
	Input :
			input_shape : shape of input images
			embeddingsize : vectorsize used to encode our picture
	'''
	inputs1 = Input(shape=(input_shape[0],), name='places')
	inputs2 = Input(shape=(input_shape[1],), name='objects')
	inputs3 = Input(shape=(input_shape[2],), name='people')


	output_pre_middle_1 = Dense(1024, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_places')(inputs1)
	output_pre_middle_2 = Dense(1024, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_objects')(inputs2)
	output_pre_middle_3 = Dense(1024, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(2e-4), name='pre_middle_people')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])
	

	output_middle = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(1e-3),kernel_initializer='he_uniform',name = 'middle_pre')(x)

	output_middle_pos = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(1e-3),kernel_initializer='he_uniform',name = 'middle')(output_middle)

	distance = Lambda(lambda x: K.l2_normalize(x,axis=-1))(output_middle_pos)

	
	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=distance)

	print(model.summary())

	return model



def build_model(input_shape, network, margin=1.0): #original 0.2
	'''
	Define the Keras Model for training
		Input :
			input_shape : shape of input images
			network : Neural network to train outputing embeddings
			margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)

	'''
	 # Define the tensors for the three input images
	anchor_input_places = Input(shape=(input_shape[0],), name="anchor_input_places")
	anchor_input_objects = Input(shape=(input_shape[1],), name="anchor_input_objects")
	anchor_input_people = Input(shape=(input_shape[2],), name="anchor_input_people")

	positive_input_places = Input(shape=(input_shape[0],), name="positive_input_places")
	positive_input_objects = Input(shape=(input_shape[1],), name="positive_input_objects")
	positive_input_people = Input(shape=(input_shape[2],), name="positive_input_people")

	negative_input_places = Input(shape=(input_shape[0],), name="negative_input_places")
	negative_input_objects = Input(shape=(input_shape[1],), name="negative_input_objects")
	negative_input_people = Input(shape=(input_shape[2],), name="negative_input_people")

	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches

	# Generate the encodings (feature vectors) for the three images
	encoded_a = network([anchor_input_places, anchor_input_objects, anchor_input_people])
	encoded_p = network([positive_input_places, positive_input_objects, positive_input_people])
	encoded_n = network([negative_input_places, negative_input_objects, negative_input_people])

	#TripletLoss Layer
	loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])

	# Connect the inputs with the outputs
	network_train = Model(inputs=[anchor_input_places,anchor_input_objects,anchor_input_people,positive_input_places,positive_input_objects,positive_input_people,negative_input_places,negative_input_objects,negative_input_people],outputs=loss_layer)

	# return the model
	return network_train


def compute_probs(network,X,Y):
	'''
	Input
		network : current NN to compute embeddings
		X : tensor of shape (m,w,h,1) containing pics to evaluate
		Y : tensor of shape (m,) containing true class

	Returns
		probs : array of shape (m,m) containing distances

	'''
	m = len(X[0])
	nbevaluation = int(m*(m-1)/2)
	probs = np.zeros((nbevaluation))
	y = np.zeros((nbevaluation))

	#Compute all embeddings for all pics with current network
	embeddings = network.predict(X)

	size_embedding = embeddings.shape[1]

	#For each pics of our dataset
	k = 0
	for i in range(m):
			#Against all other images
			for j in range(i+1,m):
				#compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
				probs[k] = -compute_dist(embeddings[i,:],embeddings[j,:])
				if (Y[i]==Y[j]):
					y[k] = 1
					#print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
				else:
					y[k] = 0
					#print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
				k += 1
	return probs,y,k


def compute_metrics(probs,yprobs):
	'''
	Returns
		fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
		tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
		thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
		auc : Area Under the ROC Curve metric
	'''
	# calculate AUC
	auc = roc_auc_score(yprobs, probs)
	# calculate roc curve
	fpr, tpr, thresholds = roc_curve(yprobs, probs)

	return fpr, tpr, thresholds,auc


def train_siamese_triplet(positive_train, negative_train, val_data, val_label, input_shape,n_val,name_path, name_final, type_net):

	print(input_shape)

	if type_net == 0:
		network = build_network(input_shape)
	elif type_net == 1:
		network = build_network_deep1(input_shape)
	elif type_net == 2:
		network = build_network_deep2(input_shape)
	else:
		network = build_network_deep3(input_shape)
	

	network_train = build_model(input_shape, network)

	adam = Adam(lr=0.001, decay=1e-5)
	network_train.compile(optimizer=adam, metrics=[accuracy])
	#change to a small batch size if necessary
	epochs = 50*(len(positive_train[0])//64) 
	print("Starting training process!")
	print("-------------------------------------")
	t_start = time.time()
	aux_sum = -1
	for i in range(1, epochs+1):
		triplets = get_batch_hard(positive_train, negative_train,512,16,16,network)
		loss = network_train.train_on_batch(triplets, None)

		print(triplets)
		print(len(triplets))
		print(len(triplets[0]))

		if i % evaluate_every == 0:
			print("\n ------------- \n")
			print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,i))
			rand = np.random.randint(0, len(val_data[0])-n_val)
			probs,yprob,k = compute_probs(network,[val_data[0][rand:rand+n_val,:],val_data[1][rand:rand+n_val,:],val_data[2][rand:rand+n_val,:]],val_label[rand:rand+n_val])
			fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
			if aux_sum < auc:
				aux_sum = auc
				network.save_weights(name_path+'/best_'+name_final)

	network.save_weights(name_path+'/'+name_final)

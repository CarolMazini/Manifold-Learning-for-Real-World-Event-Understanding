                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        #!/usr/bin/env python3
import numpy as np
import keras
from keras.models import Sequential
import keras.layers
from keras.layers import Dense, Conv2D, Flatten,MaxPooling1D,GlobalAveragePooling1D
from keras.layers import Input,concatenate
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
import h5py
from keras.callbacks import ModelCheckpoint



def define_model(input_shape, num_class):
	#create model
	model = Sequential()#add model layers
	model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(num_class, activation='softmax'))

	return model

def define_1Dmodel(input_shape,num_classes):
	model_m = Sequential()
	model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
	model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
	model_m.add(Conv1D(100, 10, activation='relu'))
	model_m.add(MaxPooling1D(3))
	model_m.add(Conv1D(160, 10, activation='relu'))
	model_m.add(Conv1D(160, 10, activation='relu'))
	model_m.add(GlobalAveragePooling1D())
	model_m.add(Dropout(0.5))
	model_m.add(Dense(num_classes, activation='softmax'))
	print(model_m.summary())
	return(model_m)

def define_neural_net(input_dim,classes):
	model = Sequential()
	input_next = input_dim//12
	model.add(Dense(input_next, input_dim=input_dim, activation='relu'))
	input_next = (input_next)//4
	model.add(Dense(input_next, activation='relu'))
	input_next = (input_next)//4
	model.add(Dense(input_next, activation='relu'))
	#model_m.add(Dropout(0.5))
	model.add(Dense(classes, activation='softmax'))
	print(model.summary())

	return model

def define_neural_net_functional(input_dim1, input_dim2, input_dim3,classes):

	inputs1 = Input(shape=(input_dim1,), name='places')
	inputs2 = Input(shape=(input_dim2,), name='objects')
	inputs3 = Input(shape=(input_dim3,), name='people')

	output_pre_middle_1 = Dense(512, activation='relu')(inputs1)
	output_pre_middle_2 = Dense(512, activation='relu')(inputs2)
	output_pre_middle_3 = Dense(512, activation='relu')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])


	output_middle = Dense(128, activation='relu')(x)


	output = Dense(classes, activation='softmax')(output_middle)

	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=output)

	print(model.summary())

	return model

#512/128/64
def define_neural_net_functional_deep1(input_dim1, input_dim2, input_dim3,classes):

	inputs1 = Input(shape=(input_dim1,), name='places')
	inputs2 = Input(shape=(input_dim2,), name='objects')
	inputs3 = Input(shape=(input_dim3,), name='people')

	output_pre_middle_1 = Dense(512, activation='relu')(inputs1)
	output_pre_middle_2 = Dense(512, activation='relu')(inputs2)
	output_pre_middle_3 = Dense(512, activation='relu')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])


	output_middle = Dense(128, activation='relu')(x)

	output_pos_middle = Dense(64, activation='relu')(output_middle)

	output = Dense(classes, activation='softmax')(output_pos_middle)

	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=output)

	print(model.summary())

	return model

#1024/512
def define_neural_net_functional_deep2(input_dim1, input_dim2, input_dim3,classes):

	inputs1 = Input(shape=(input_dim1,), name='places')
	inputs2 = Input(shape=(input_dim2,), name='objects')
	inputs3 = Input(shape=(input_dim3,), name='people')

	output_pre_middle_1 = Dense(1024, activation='relu')(inputs1)
	output_pre_middle_2 = Dense(1024, activation='relu')(inputs2)
	output_pre_middle_3 = Dense(1024, activation='relu')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])


	output_middle = Dense(512, activation='relu')(x)


	output = Dense(classes, activation='softmax')(output_middle)

	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=output)

	print(model.summary())

	return model

#1024/512/128
def define_neural_net_functional_deep3(input_dim1, input_dim2, input_dim3,classes):

	inputs1 = Input(shape=(input_dim1,), name='places')
	inputs2 = Input(shape=(input_dim2,), name='objects')
	inputs3 = Input(shape=(input_dim3,), name='people')

	output_pre_middle_1 = Dense(1024, activation='relu')(inputs1)
	output_pre_middle_2 = Dense(1024, activation='relu')(inputs2)
	output_pre_middle_3 = Dense(1024, activation='relu')(inputs3)

	x = concatenate([output_pre_middle_1,output_pre_middle_2, output_pre_middle_3])


	output_middle = Dense(512, activation='relu')(x)

	output_pos_middle = Dense(128, activation='relu')(output_middle)
	
	output = Dense(classes, activation='softmax')(output_pos_middle)

	model = Model(inputs=[inputs1, inputs2,inputs3], outputs=output)

	print(model.summary())

	return model


def define_neural_net_siamese(input_dim1, input_dim2, input_dim3,classes):

	inputs1_a = Input(shape=(input_dim1,), name='places_a')
	inputs2_a = Input(shape=(input_dim2,), name='objects_a')
	inputs3_a = Input(shape=(input_dim3,), name='people_a')

	inputs1_b = Input(shape=(input_dim1,), name='places_b')
	inputs2_b = Input(shape=(input_dim2,), name='objects_b')
	inputs3_b = Input(shape=(input_dim3,), name='people_b')


	dense_pre_middle_1 = Dense(512, activation='relu')
	dense_pre_middle_2 = Dense(512, activation='relu')
	dense_pre_middle_3 = Dense(512, activation='relu')

	dense_middle = Dense(128, activation='relu')

	output_pre_middle_1a =dense_pre_middle_1(inputs1_a)
	output_pre_middle_2a = dense_pre_middle_2(inputs2_a)
	output_pre_middle_3a = dense_pre_middle_3(inputs3_a)

	output_pre_middle_1b = dense_pre_middle_1(inputs1_b)
	output_pre_middle_2b = dense_pre_middle_2(inputs2_b)
	output_pre_middle_3b = dense_pre_middle_3(inputs3_b)

	x = concatenate([output_pre_middle_1a,output_pre_middle_2a, output_pre_middle_3a])
	y = concatenate([output_pre_middle_1b,output_pre_middle_2b, output_pre_middle_3b])


	output_middle_x = dense_middle(x)
	output_middle_y = dense_middle(y)

	merged_vector = concatenate([output_middle_x, output_middle_y], axis=-1)



	final_dense = Dense(classes, activation='softmax')

	output = final_dense(merged_vector)

	model = Model(inputs=[inputs1_a, inputs2_a,inputs3_a,inputs1_b, inputs2_b,inputs3_b], outputs=output)

	#assert dense_middle.get_output_at(0) == output_middle_x
	#assert dense_middle.get_output_at(1) == output_middle_y

	print(model.summary())

	return model


def data_generator(positive_data,negative_data, batch_size = 64):

	while True:
		# Select files (paths/indices) for the batch
		batch_paths_positive = np.random.choice(a = len(positive_data), size = batch_size//2,replace=True)
		batch_paths_negative = np.random.choice(a = len(negative_data), size = batch_size//2,replace=True)
		batch_input1 = []
		batch_input2 = []
		batch_input3 = []
		batch_output = []

		for index in range(batch_size//2):

			batch_input1.append(positive_data[0][batch_paths_positive[index]])
			batch_input2.append(positive_data[1][batch_paths_positive[index]])
			batch_input3.append(positive_data[2][batch_paths_positive[index]])
			batch_output.append([0,1])

			batch_input1.append(negative_data[0][batch_paths_negative[index]])
			batch_input2.append(negative_data[1][batch_paths_negative[index]])
			batch_input3.append(negative_data[2][batch_paths_negative[index]])
			batch_output.append([1,0])


		# Return a tuple of (input,output) to feed the network

		batch_x1 = np.array(batch_input1)
		batch_x2 = np.array(batch_input2)
		batch_x3 = np.array(batch_input3)
		batch_y = np.array(batch_output)
		#print(batch_input1)

		yield({'places': batch_x1, 'objects': batch_x2, 'people': batch_x3}, batch_y)


def train_model(model,train_positive_data,train_negative_data,val_positive_data,val_negative_data,epochs,name_path):

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

	steps_training = (len(train_positive_data[0])+len(train_negative_data[0]))//32

	steps_validation = (len(val_positive_data[0])+len(val_negative_data[0]))//6


	history = model.fit_generator(data_generator(train_positive_data,train_negative_data, batch_size = 32),
						epochs=epochs,
						steps_per_epoch = steps_training ,
						validation_steps = steps_validation,
						# We pass some validation for
						# monitoring validation loss and metrics
						# at the end of each epoch
						validation_data=data_generator(val_positive_data,val_negative_data, batch_size = 6),
						callbacks=[ModelCheckpoint(filepath=name_path, save_best_only=True,save_weights_only=True)])

	# The returned "history" object holds a record
	# of the loss values and metric values during training
	print('\nhistory dict:', history.history)
	return model


def evaluate_model(model,test_data,label_data):
	# Evaluate the model on the test data using `evaluate`
	print('\n# Evaluate on test data')
	results = model.evaluate(test_data, label_data, batch_size=128)
	print('test loss, test acc:', results)

	# Generate predictions (probabilities -- the output of the last layer)
	# on new data using `predict`
	print('\n# Generate predictions for 3 samples')
	predictions = model.predict(test_data)
	print('predictions shape:', predictions.shape)


def extract_features(model,filepath):

    model.load_weights(filepath)

    # Freeze the layers except the last train_layers
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers:
        print(layer, layer.trainable)

    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    return model

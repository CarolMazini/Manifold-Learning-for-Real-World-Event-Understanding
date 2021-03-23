from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import random
import time
import argparse


##################################MAIN#####################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Training SVM classifier with combined features.')
	parser.add_argument('--dataset', dest='dataset', type=str,help='include the name of the dataset to extract image features',default='bombing')
	parser.add_argument('--method', dest='method', type=str,help='type of train combination: contrastive or triplet',default='triplet')
	parser.add_argument('--aug', dest='aug', type=str,help='include _aug to use augmented data', default='')
	args = parser.parse_args()
       
	dataset = args.dataset #'wedding', 'fire', 'bombing', 'museu_nacional' or 'bangladesh_fire'
	aug=args.aug
	method = args.method
	random.seed(a=0)


	def compute_normalized_accuracy(y_true, y_pred):

		correct_pos = 0
		correct_neg = 0
		incorrect_pos = 0
		incorrect_neg = 0
		
		total_pos = np.sum(y_true)
		total_neg = len(y_true) - total_pos
		
		for i in range(len(y_pred)):
			if y_true[i] == 1:
				if y_pred[i] == 1:
					correct_pos+=1
				else:
					incorrect_pos+=1
			else:
				if y_pred[i] == 0:
					correct_neg+=1
				else:
					incorrect_neg+=1
		norm_acc = (correct_pos/total_pos + correct_neg/total_neg)/2.0
		precision = correct_pos/(correct_pos+incorrect_neg+0.000001)
		recall = correct_pos/(total_pos+0.000001)
		
		print('correct pos: ', correct_pos)	
		print('correct neg: ', correct_neg)
		print('incorrect pos: ', incorrect_pos)
		print('incorrect pos: ', incorrect_neg)		
					
		return norm_acc,precision,recall


	X_train_positive = np.load('../out_files/features/'+method+'/'+dataset+'/1024_512_positive_'+dataset+aug+'_train.npy')
	X_train_negative = np.load('../out_files/features/'+method+'/'+dataset+'/1024_512_negative_'+dataset+aug+'_train.npy')

	Y_train_positive = np.ones(len(X_train_positive))
	Y_train_negative = np.zeros(len(X_train_negative))

	X_train = np.concatenate([X_train_positive, X_train_negative], axis=0)
	Y_train = np.concatenate([Y_train_positive, Y_train_negative], axis=0)

	total_train = len(X_train)

	sampled_list = random.sample(range(total_train), total_train)
	X_train = X_train[sampled_list,:]
	Y_train = Y_train[sampled_list]

	X_test_positive = np.load('../out_files/features/'+method+'/'+dataset+'/1024_512_positive_'+dataset+aug+'_test.npy')
	X_test_negative = np.load('../out_files/features/'+method+'/'+dataset+'/1024_512_negative_'+dataset+aug+'_test.npy')

	Y_test_positive = np.ones(len(X_test_positive))
	Y_test_negative = np.zeros(len(X_test_negative))

	X_test = np.concatenate([X_test_positive, X_test_negative], axis=0)
	Y_test = np.concatenate([Y_test_positive, Y_test_negative], axis=0)

	print('Total pos: ', X_test_positive.shape)
	print('Total neg: ', len(X_test_negative))

	start_time = time.time()


	clf = make_pipeline(StandardScaler(), SVC(kernel = 'rbf',C=1.0,gamma='auto'))
	clf.fit(X_train, Y_train)

	y_pred =clf.predict(X_test) #classe

	print("--- %s seconds ---" % (time.time() - start_time))

	norm_acc,precision,recall = compute_normalized_accuracy(Y_test, y_pred)

	print('Norm Acc: ', norm_acc)
	print('Precision: ', precision)
	print('Recall: ', recall)

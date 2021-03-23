
import matplotlib.pyplot as plt
import numpy as np
import math
import shutil
import os,sys
import glob
from PIL import Image, ImageOps


def plotImages(paths, labels,ncols,nrows,nameBaseN, nameBaseP,k_ini,consulta):

	#fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
	fig = plt.figure()
	axeslist = [ fig.add_subplot(nrows, ncols, r * ncols + c+1) for r in range(0, nrows) for c in range(0, ncols) ]
	print(len(paths))
	
	#for axi in axeslist.ravel():	
	#	axi.axis('off')
	for i in range(1,ncols+1):
		fig.add_subplot(nrows, ncols, i)
		img = Image.open(nameBaseP+consulta)
		# img = ImageOps.expand(img,border=30,fill='black')
		plt.imshow(img,aspect='auto')

	for i in range(ncols+1, ncols*nrows +1):
		#img = imread(paths[i])
		fig.add_subplot(nrows, ncols, i)
		
		if(labels[i-1] == 0):
			img = Image.open(nameBaseN+paths[i-1])
			#img = ImageOps.expand(img,border=30,fill='red')
		else:
			img = Image.open(nameBaseP+paths[i-1])
			#img = ImageOps.expand(img,border=30,fill='green')
		#axeslist[i//ncols - 1, i%ncols].axis('off')
		plt.imshow(img,aspect='auto')
	
	plt.tick_params(
		axis='x',		  # changes apply to the x-axis
		which='both',	  # both major and minor ticks are affected
		bottom=False,	  # ticks along the bottom edge are off
		top=False,		 # ticks along the top edge are off
		labelbottom=False) # labels along the bottom edge are off
	plt.tick_params(
		axis='y',		  # changes apply to the x-axis
		which='both',	  # both major and minor ticks are affected
		bottom=False,	  # ticks along the bottom edge are off
		top=False,		 # ticks along the top edge are off
		labelbottom=False) # labels along the bottom edge are off
	for ax in axeslist:
		ax.set_xticks([])
		ax.set_yticks([])


	#plt.title("Analysis per Rank (K = %d, Rankings = %d)" %(nrows,ncols))
	plt.savefig(dataset+"_rankAnalysis_teste_borda_"+str(k_ini)+".png", bbox_inches='tight')
	plt.show()


def getFilenames(ranking, filenames, k_ini,k, relevantNum):

	names = []
	labels = []

	for i in range(k_ini, k):
		names.append(filenames[ranking[i]])
		if ranking[i]< relevantNum:
			labels.append(1)
		else:
			labels.append(0)

	return names, labels




##################################MAIN#####################################

if __name__ == '__main__':

	np.set_printoptions(threshold=np.inf)
	
	dataset = 'bombing' #'wedding', 'fire', museu_nacional or bangladesh_fire
	path_data = os.path.join('../dataset',dataset)
	path_names = os.path.join('../out_files','features')
	base_ranking = os.path.join('../out_files','ranking')

	relevant_names =  np.load(os.path.join(path_names,dataset+'/positive_test_names_places.npy'))
	not_names = np.load(os.path.join(path_names,dataset+'/negative_test_names_places.npy'))

	nameBaseN = os.path.join(path_data,'negative/')
	nameBaseP = os.path.join(path_data,'positive/')


	relevantNum = len(relevant_names)

	filenames = np.concatenate([relevant_names, not_names], axis=0)

	consulta_names = np.load(os.path.join(path_names,dataset+'/positivetrain_names_places.npy'))

	sortedIndicesConcat = np.load(os.path.join(base_ranking,dataset,'rankings_relevant_concatenated.npy'))[0]
	sortedIndicesESS = np.load(os.path.join(base_ranking,'ess',dataset,'rankings_relevant_ess.npy'))[0]
	sortedIndicesFine = np.load(os.path.join(base_ranking,'fine_tunned',dataset,'rankings_relevant_finetuned.npy'))[0]
	sortedIndicesCross = np.load(os.path.join(base_ranking,'cross_entropy',dataset,'rankings_relevant_cross.npy'))[0]
	sortedIndicesContrastive = np.load(os.path.join(base_ranking,'contrastive',dataset,'rankings_relevant_contrastive.npy'))[0]
	sortedIndicesTriplet = np.load(os.path.join(base_ranking,'triplet',dataset,'rankings_relevant_triplet.npy'))[0]
	

	nrows = 6
	ncols = 6
	k_ini = 0

	for j in range(0,10):

		finalNames = []
		finalLabel = []


		names1, labels1 = getFilenames(sortedIndicesConcat, filenames, k_ini, k_ini+nrows, relevantNum)
		names2, labels2 = getFilenames(sortedIndicesESS, filenames, k_ini, k_ini+nrows, relevantNum)
		names3, labels3 = getFilenames(sortedIndicesFine, filenames, k_ini, k_ini+nrows, relevantNum)
		names4, labels4 = getFilenames(sortedIndicesCross, filenames, k_ini, k_ini+nrows, relevantNum)
		names5, labels5 = getFilenames(sortedIndicesContrastive, filenames, k_ini, k_ini+nrows, relevantNum)
		names6, labels6 = getFilenames(sortedIndicesTriplet, filenames, k_ini, k_ini+nrows, relevantNum)

		for i in range(0, nrows):
			finalNames.append(names1[i])
			finalNames.append(names2[i])
			finalNames.append(names3[i])
			finalNames.append(names4[i])
			finalNames.append(names5[i])
			finalNames.append(names6[i])
			finalLabel.append(labels1[i])
			finalLabel.append(labels2[i])
			finalLabel.append(labels3[i])
			finalLabel.append(labels4[i])
			finalLabel.append(labels5[i])
			finalLabel.append(labels6[i])


		plotImages(finalNames, finalLabel,ncols,nrows,nameBaseN,nameBaseP, k_ini, consulta_names[0])

		k_ini = k_ini +5

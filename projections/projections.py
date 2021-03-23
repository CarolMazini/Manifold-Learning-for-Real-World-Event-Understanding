#!/usr/bin/env python

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from sklearn.datasets import make_blobs
import random

colors = ['orchid','darkblue','royalblue','r','g','b', 'c', 'm', 'y', 'darksalmon', 'blueviolet','black','darkred','silver']


def choice_color(number_of_colors):

	color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

	return(color)

def scatter_plot_data(data, labels, subtitles, num_class,save_path):

	fig, ax = plt.subplots(figsize=(4, 4))


	cores4 = ['#ad7a99','#7a94ae','#62c9bf','#27726a']
	cores = ['#62c9bf', '#ad7a99']

	plt.rcParams.update({'font.size': 22}) #16,20
	print(data)
	for i,c in enumerate(cores4):

		mask = labels[:,1] == i+1

		print(labels[mask])
		cor = np.full(len(labels[mask]),c)

		ax.scatter(data[mask,0],data[mask,1],data[mask,2], marker='o', color=cores4[i], facecolors='None') #,label = subtitles[i]


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
	ax.set_xticks([])
	ax.set_yticks([])

	ax.grid(False)
    
	plt.savefig(save_path)

def scatter_plot_data_by_distance(data, labels, subtitles, num_class,save_path):
	
	fig, ax = plt.subplots(figsize=(4, 4))


	cores4 = ['#ad7a99','#7a94ae','#62c9bf','#27726a']
	cores = ['#62c9bf', '#ad7a99']
	plt.rcParams.update({'font.size': 22}) #16,20

	index = np.argsort(-data[:,2])

	print(index)

	for i in index:

		ax.scatter(data[i,0],data[i,1], marker='o', color=cores[int(labels[i,0])], facecolors='None') #,label = subtitles[i]



	plt.tick_params(
				axis='x',		 # changes apply to the x-axis
				which='both',	 # both major and minor ticks are affected
				bottom=False,	 # ticks along the bottom edge are off
				top=False,	   # ticks along the top edge are off
				labelbottom=False) # labels along the bottom edge are off
	plt.tick_params(
				axis='y',		 # changes apply to the x-axis
				which='both',	 # both major and minor ticks are affected
				bottom=False,	 # ticks along the bottom edge are off
				top=False,	   # ticks along the top edge are off
				labelbottom=False) # labels along the bottom edge are off
	ax.set_xticks([])
	ax.set_yticks([])

	ax.grid(False)
	



	plt.savefig(save_path)

	### 4 classes

	fig, ax = plt.subplots(figsize=(4, 4))

	for i in index:

		ax.scatter(data[i,0],data[i,1], marker='o', color=cores4[int(labels[i,1]-1)], facecolors='None') #,label = subtitles[i]



	plt.tick_params(
				axis='x',		 # changes apply to the x-axis
				which='both',	 # both major and minor ticks are affected
				bottom=False,	 # ticks along the bottom edge are off
				top=False,	   # ticks along the top edge are off
				labelbottom=False) # labels along the bottom edge are off
	plt.tick_params(
				axis='y',		 # changes apply to the x-axis
				which='both',	 # both major and minor ticks are affected
				bottom=False,	 # ticks along the bottom edge are off
				top=False,	   # ticks along the top edge are off
				labelbottom=False) # labels along the bottom edge are off
	ax.set_xticks([])
	ax.set_yticks([])

	ax.grid(False)
	plt.savefig(save_path.split('.png')[-2]+'_multiclass.png')

def scatter_plot_data_alternate(data, labels, subtitles, num_class,save_path):
	fig, ax = plt.subplots(figsize=(4, 4))	
	cores = ['#62c9bf', '#ad7a99']
	plt.rcParams.update({'font.size': 22}) #16,20
	print(data)

	random.seed(a=42)

	#sampling train and val sets
	index = random.sample(range(0,len(data)), len(data))

	for i in index:

		ax.scatter(data[i,0],data[i,1], marker='o', color=cores[int(labels[i])], facecolors='None') #,label = subtitles[i]


	plt.tick_params(
				axis='x',		 # changes apply to the x-axis
				which='both',	 # both major and minor ticks are affected
				bottom=False,	 # ticks along the bottom edge are off
				top=False,	   # ticks along the top edge are off
				labelbottom=False) # labels along the bottom edge are off
	plt.tick_params(
				axis='y',		 # changes apply to the x-axis
				which='both',	 # both major and minor ticks are affected
				bottom=False,	 # ticks along the bottom edge are off
				top=False,	   # ticks along the top edge are off
				labelbottom=False) # labels along the bottom edge are off
	ax.set_xticks([])
	ax.set_yticks([])

	ax.grid(False)
	#ax.legend(False)
	#ax.legend(loc='upper left')
	#ax.legend(loc='best')
	#ax.set_title('Adapted')


	plt.savefig(save_path)


def line_simple_plot(data_y,x, num_lines, markers,x_label,y_label, subtitles,path_out,cores,title):

	fig, ax = plt.subplots(figsize=(8, 4))
	#fig, ax = plt.subplots(figsize=(8, 5))
	plt.gcf().subplots_adjust(top=0.8) #0.75
	#plt.gcf().subplots_adjust(top=0.8)
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.gcf().subplots_adjust(left=0.1)
	plt.rc('xtick', labelsize=14) #14,16
	plt.rc('ytick', labelsize=14) #14,16
	plt.rcParams.update({'font.size': 14}) #16,20
	#plt.gcf().subplots_adjust(bottom=0.2) 

	for i in range(num_lines):
		print(data_y[i])
		ax.plot(x,data_y[i],label=subtitles[i],marker = markers[i],c=cores[i],linewidth=2, markersize=14,alpha = 1.0) #2,14,4,16


	ax.grid(color='grey', alpha=0.2)
	ax.legend(loc='right')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),fancybox=True, shadow=False, ncol=4) #1.45
	#ax.set_title(title)
	#ax.get_legend().remove()
	#ax.spines['top'].set_visible(False)
	#ax.spines['right'].set_visible(False)



	plt.savefig(path_out)


def line_error_plot(media_y,x,var_y, num_lines, markers,x_label,y_label, subtitles,path_out,cores,title):



	fig, ax = plt.subplots(figsize=(6, 4))
	plt.gcf().subplots_adjust(top=0.75)
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.gcf().subplots_adjust(left=0.1)
	#plt.rc('xtick', labelsize=16) #14,16
	#plt.rc('ytick', labelsize=16) #14,16
	plt.rcParams.update({'font.size': 9}) #16,20
	ax.tick_params(axis='both', which='major', labelsize=10)


	label_y = ('0.0','0.2','0.4','0.6','0.8','1.0')
	#plt.xticks(np.arange(0, 501, 100))

	for i in range(num_lines):
		print(media_y[i])

		#plt.yticks(np.arange(0.0, 1.2,step=0.05))
		ax.errorbar(x,media_y[i],yerr=var_y[i],label=subtitles[i],marker = markers[i],c=cores[i],linewidth=2, markersize=10,alpha = 1.0)



	ax.grid(color='grey', alpha=0.2)
	ax.legend(loc='right')
	plt.xlabel(x_label, size=10)
	plt.ylabel(y_label, size=10)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.32),fancybox=True, shadow=False, ncol=4) #1.45
	#ax.set_title(title)
	#ax.get_legend().remove()
	#ax.spines['top'].set_visible(False)
	#ax.spines['right'].set_visible(False)

	plt.savefig(path_out)

#!/usr/bin/env python3
import numpy as np
import math


def create_ess_representation(distances, num_descriptors, seeds):

	finalVec = np.full((len(distances[0]),num_descriptors*len(seeds)), np.inf)

	for j in range(len(seeds)):
		for i in range(len(finalVec)):
			for k in range(num_descriptors):
				finalVec[i,num_descriptors*j+k] = distances[k][i,j]


	return finalVec

    
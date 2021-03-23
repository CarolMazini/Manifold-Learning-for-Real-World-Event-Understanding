#!/usr/bin/env python3
import numpy as np
import math
from sklearn.decomposition import PCA
import os
import glob
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler,normalize,Normalizer


def pca_transform(data):

    pca = PCA()

    completeNorm = Normalizer().fit_transform(data)

    completeNorm = pca.fit_transform(completeNorm)

    return completeNorm

def tsne_transform(data,components):

    completeNorm = TSNE(n_components=components).fit_transform(data)

    return completeNorm

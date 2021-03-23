#!/usr/bin/env python3
import numpy as np
import math

##correlation metric for cost function
def jaccard(i1, i2,topK):
    """
    entrada: dois ranking i1 e i2, um valor de top até o qual os rankings serão comparados
    saida: indice de jaccard, 0 indica que não existe intersecao entre os rankings, 1 indica intersecao completa

    o indice de jaccard é calculado com intersecao/unicao dos rankings
    """
    

    setI1 = set(i1[0:topK])
    setI2 = set(i2[0:topK])

    inter = setI1.intersection(setI2)
    uni = setI1.union(setI2)


    #Calculate Jaccard Index
    jaccard = len(inter)/len(uni)

    return jaccard


def inverse_inter(i1, i2,topK):
    """
    entrada: dois ranking i1 e i2, um valor de top até o qual os rankings serão comparados
    saida: inverso do tamanho da intersecao, 0 indica que houve interesecao completa, topK indica que a intersecao é vazia

    o indice de jaccard é calculado com intersecao/unicao dos rankings
    """
    

    setI1 = set(i1[0:topK])
    setI2 = set(i2[0:topK])

    inter = setI1.intersection(setI2)

    #Calculate Jaccard Index
    inverse = topK - len(inter)

    return inverse

def correlacao_ranking_jaccard(rankings, topK):

    """
    entrada: matriz de rankings onde cada linha é um ranking, número do top a ser analisado
    saida: soma normalizada dos indices de jaccard dos rankings dois a dois

    utiliza a funcao jaccard(i1, i2,topK) para calculo do indice
    """

    sum_jaccard = 0
    num = 0
    for i in range(len(rankings) - 1):
        for j in range(i+1,len(rankings)):
            jac = jaccard(rankings[i],rankings[j],topK)
            num+=1
            sum_jaccard += jac
    return (sum_jaccard/num) 

##distance metric for cost function
def distancia_ranking(scores,samples):

    """
    entrada: uma matriz de scores onde cada coluna é composta por distancias das amostras para uma semente (ranking não ordenado), e um conjunto de samples novas que são candidatas a solução que deseja-se saber a distancia para as sementes
    saida: soma normalizada das distancias de cada semente candidata para cada semente do ranking original
    """

    sum_scores = 0
    num = 0

    for j in range(len(scores[0])):
        for i in samples:
            sum_scores += scores[int(i),j] 
            num+=1

    return sum_scores/num

def weighted_position_ranking(top, ranking, j):

    """
    entrada: um ranking (da imagem q), o tamanho do top, e uma imagem j para ser obtido o peso
    saida: peso da imagem j
    """

    position = -1
    for i in range(top):
        if ranking[i] == j:
            position = i
            break

    if position == -1:
        weight = 0
    else:
        weight = top - position

    return weight

def reciprocal_neighborhood_density(q,i,ordenado,top):

    """
    entrada: uma matriz de posicoes, duas imagens q e i para serem avaliadas, o tamanho do topo
    saida: medida de estimativa de eficacia 
    """
    reciprocal = 0
    eficacia = 0

    for j in ordenado[0:top,q]:
        for l in ordenado[0:top,i]:
            reciprocal = 0
            if (j in ordenado[0:top,l]) and (l in ordenado[0:top,j]):
                reciprocal = 1
            eficacia+=reciprocal*weighted_position_ranking(top, ordenado[:,q], j)*weighted_position_ranking(top, ordenado[:,i], l)

    return eficacia/(top*top*top*top)


def authority(q,ordenado,top):

    """
    entrada: uma matriz de posicoes, duas imagens q e i para serem avaliadas, o tamanho do topo
    saida: medida de estimativa de eficacia 
    """

    eficacia = 0.0

    for i in ordenado[0:top,q]:
        for j in ordenado[0:top,i]:
            if (j in ordenado[0:top,q]):
                eficacia+=1.0

    return eficacia/(top*top)

def weighted_log(top, ranking, j):
    """
    entrada: um ranking (da imagem q), o tamanho do top, e uma imagem j para ser obtido o peso
    saida: peso da imagem j
    """

    weight = 0
    for i in range(top):
        if ranking[i] == j:
            weight = 1 - math.log(i+1,top)
            break

    return weight

def hyperedge_weight(i,j,ordenado, top):
    """
    entrada: uma matriz de posicoes, duas imagens i e x para serem avaliadas, o tamanho do topo
    saida: peso da hyperaresta 
    """

    peso = 0.0

    for x in ordenado[0:top,i]:
        if j in ordenado[0:top,x]:
            peso+=weighted_log(top, ordenado[:,i], x)*weighted_log(top, ordenado[:,x], j)

    return peso

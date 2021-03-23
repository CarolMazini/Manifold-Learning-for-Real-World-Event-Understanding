#!/usr/bin/env python3
import numpy as np
import math
from sklearn.metrics import pairwise_distances 




##distance metrics calculus
def distCalculo(X,seeds, metric = 'euclidean'):
    """
    entrada: uma matriz de features onde as linhas representam as amostras e as colunas sao os atributos, um vetor com indices das sementes na matriz de features, a metrica a ser utilizada
    saida: as distancias de cada amostra para cada semente onde a linha i sao todas as distancias da amostra i para a semente e a coluna j sao todas as distancias das amostras para a semente j 
    (os rankings sao obtidos ordenando cada coluna)

    usa a funcao paiwise_distance do sklearn
    """
    ideal = []
    if seeds  is None:
        ideal = None
    elif seeds ==[]:
        ideal.append(np.zeros(len(X[0])))
    else:    
        for i in seeds:
            ideal.append(X[int(i)])

    sumAll = pairwise_distances(X, Y=ideal, metric=metric)
    #sumAll = sumAll.reshape(1,len(sumAll))[0]

    return sumAll

def select_distances_by_index(X,index):
    """
    entrada: uma matriz de distancias e os indices das colunas que desejamos selecionar
    saida: uma matriz de distancias dos indices especificados (colunas)
    """

    X_final = np.empty((len(X),len(index)))
    
    for i in range(len(index)):
        X_final[:,i] = X[:,int(index[i])]

    return X_final 


##calculating index of recall points
def calculateIndexRecall(sortedIndicesFinal,relevantNum):    

    """
    entrada: ranking como vetor de posicoes (ja ordenado por menor distancia) e numero de imagens relevantes (indices ate aquele numero sao relevantes)
    saida: vetor ground truth onde 1 representa imagens relevantes e 0 imagens nao relevantes, e vetor com a posicao do ranking onde e possivel recuperar cada uma dos 10 pontos de recall (porcentagem de 10 em 10 ate 100% recall)

    """

    indexOurs = 0
    accOurs = 0
    recallIndexOurs = np.full(10, np.inf)
    binaryOurs= np.full(len(sortedIndicesFinal), np.inf)
    recallNumTotal = np.full(10, np.inf)
   
    for i in range(0,10):
        recallNumTotal[i]= int((0.1*(i+1))*relevantNum)


    for i in range(10):
        recallIndexOurs[i] = len(sortedIndicesFinal)



    for i in range(0,len(sortedIndicesFinal)):

        if(indexOurs<10 and sortedIndicesFinal[i] <relevantNum):

            binaryOurs[i]=1
            accOurs = accOurs+1
            

            while(indexOurs<10) and (accOurs >= recallNumTotal[indexOurs]):
                recallIndexOurs[indexOurs] = i
                indexOurs = indexOurs +1

        else:
            binaryOurs[i]=0

    return binaryOurs, recallIndexOurs

def calculateIndexRecallbyVector(sortedIndicesFinal,relevant):    

    """
    entrada: ranking como vetor de posicoes (ja ordenado por menor distancia) e numero de imagens relevantes (indices ate aquele numero sao relevantes)
    saida: vetor ground truth onde 1 representa imagens relevantes e 0 imagens nao relevantes, e vetor com a posicao do ranking onde e possivel recuperar cada uma dos 10 pontos de recall (porcentagem de 10 em 10 ate 100% recall)

    """

    tam_relevant = len(relevant[relevant > 0])
    #print(tam_relevant)

    indexOurs = 0
    accOurs = 0
    recallIndexOurs = np.full(10, np.inf)
    binaryOurs= np.full(len(sortedIndicesFinal), np.inf)
    recallNumTotal = np.full(10, np.inf)
   
    for i in range(0,10):
        recallNumTotal[i]= int((0.1*(i+1))*tam_relevant)


    for i in range(10):
        recallIndexOurs[i] = len(sortedIndicesFinal)



    for i in range(0,len(sortedIndicesFinal)):

        if(indexOurs<10 and sortedIndicesFinal[i] in relevant):

            binaryOurs[i]=1
            accOurs = accOurs+1
            

            while(indexOurs<10) and (accOurs >= recallNumTotal[indexOurs]):
                recallIndexOurs[indexOurs] = i
                indexOurs = indexOurs +1

        else:
            binaryOurs[i]=0

    return binaryOurs, recallIndexOurs

##obtaining index of seeds
def seedsIndex(filename, filenameSeeds):
    """
    entrada: duas listas de nomes uma com os nomes relevantes e outra com os nomes das seeds
    saida: indice das seeds na lista de relevantes
    """
    index = []
    for name in filenameSeeds:
        aux = -1
        for i in range(0, len(filename)):
            if(name.split('_')[-1] == 'frame'+filename[i].split('_')[-1]+'.jpg') and (name.split('/')[-1].split('_')[0] == filename[i].split('_')[0]):
                aux=1
                index.append(i)
                break
        if(aux==-1):
            print(name)
            print("Seed not in relevant set!")

    return(index)

##ordering distances to construct ranking
def calcula_posicao_ranking(distancias):
    """
    entrada: matriz de distancias onde cada ranking nao ordenado e uma coluna
    saida: matriz de posicoes onde cada coluna e um ranking ordenado onde o valor de cada posicao e um indice para o vetor de distancias (indice na coluna correspondente da semente)
    """
    posicoes = []
    for i in range(len(distancias[0])):
        temp = np.argsort(distancias[:,i])
        posicoes.append(np.copy(temp))

    posicoes = np.transpose(np.array(posicoes))

    return posicoes

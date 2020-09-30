import numpy as np
import itertools as it

def d(x,y):
    return np.sqrt(np.dot(x-y,x-y))

def distance_matrix(data_matrix):
    N = data_matrix.shape[0]
    D = np.zeros([N,N])
    for (i,j) in it.product(range(N),range(N)):
        D[i,j]= d(data_matrix[i,:],data_matrix[j,:])
    return D

def density_filtration(data_matrix,k):
    D = distance_matrix(data_matrix)
    N = data_matrix.shape[0]
    dim = data_matrix.shape[1]
    filtration_matrix = np.zeros([N,dim+1])
    filtration_matrix[:,:dim]=data_matrix
    for i in range(N):
        row = data_matrix[i,:]
        k_nearest = sorted(row,reverse=True)[:k]
        summation = 0
        for j,x in enumerate(k_nearest):
            summation+=D[i,j]
        filtration_matrix[i,dim]=summation/k
    return filtration_matrix

def curvature_filtration(data_matrix,k):
    return None
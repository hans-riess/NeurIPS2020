import numpy as np
import numpy.random as rd
import itertools as it

def normalize_array(matrix,normalize='L1'):
    for i,row in enumerate(matrix):
        if normalize == 'L2':
            matrix[i,:]=row/np.linalg.norm(row)
        elif normalize == "L1":
            matrix[i,:]=row/np.sum(row)
        elif normalize == "stochastic":
            matrix[i,:]=row/np.max(row)
    return matrix

def p_convolve(conv_filter,signal,filter_pos=(0,0)):
    (m,n)=signal.shape
    (m_f,n_f)=conv_filter.shape
    (i0,j0)=filter_pos
    output = np.zeros([m,n])
    if m_f<m and n_f<n:
        result = np.zeros([m,n])
        result[i0:m_f+i0,j0:n_f+j0]=conv_filter
        conv_filter = result
    elif conv_filter.shape!=signal.shape:
        print('Filter size exceeds the dimension of the signal array')
        return None
    for x in it.product(range(m),range(n)):
        terms = []
        for e in it.product(range(m),range(n)):
            term = conv_filter[e]*signal[min(e,x)]
            terms.append(term)
        output[x] = np.sum(terms)
    return output

def local_max_pool(signal):
    (m,n)=signal.shape
    output = np.zeros([m,n])
    for (i,j) in it.product(range(m),range(n)):
        output[i,j]= np.max([signal[i,j],signal[min(i+1,m-1),j],signal[max(i-1,0),j],signal[i,min(j+1,m-1)],signal[i,max(j-1,0)]])
    return output

# def p_translate_convolve(conv_filter,signal):
#     (m,n)=signal.shape
#     (m_f,n_f)=conv_filter.shape
#     output = np.zeros([m,n])
#     for i0 in range(m-m_f+1):
#         for j0 in range(n-n_f+1):
#             output+=p_convolve(conv_filter,signal,(i0,j0))
#     return output


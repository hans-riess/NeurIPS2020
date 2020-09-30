import numpy as np
import torch
from puschel import normalize_array,p_convolve,local_max_pool

def sigmoid(x):
    return 1/(1+np.e**(-x))

def relu(x):
    return np.maximum(np.zeros(x.shape),x)





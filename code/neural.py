import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


def build_meet_convolve_tensor(signal_dim,kernel_dim,kernel_loc):
  # produces a tensor M_ijk such that the contraction M_ijk f_i is equal to f_{j ^ k}
  M = torch.zeros(signal_dim,signal_dim,kernel_dim)
  # this is an inefficient implementation, but should be fast enough for the sizes we're doing.
  for i in range(signal_dim):
    for j in range(i,signal_dim):
      for k in range(kernel_dim):
        if i == min(j,k+kernel_loc):
          M[i,j,k] = 1
  return M

def build_join_convolve_tensor(signal_dim,kernel_dim,kernel_loc):
  # produces a tensor M_ijk such that the contraction M_ijk f_i is equal to f_{j v k}
  M = torch.zeros(signal_dim,signal_dim,kernel_dim)
  for i in range(signal_dim):
    for j in range(i,signal_dim):
      for k in range(kernel_dim):
        if i == max(j,k+kernel_loc):
          M[i,j,k] = 1
  return M

class MeetConv2d(nn.Module):
  def __init__(self,signal_dim,kernel_dim,kernel_loc,in_features,out_features):
    super(MeetConv2d,self).__init__()
    (signal_x, signal_y) = signal_dim
    (kernel_x, kernel_y) = kernel_dim
    (loc_x, loc_y) = kernel_loc
    conv_x = build_meet_convolve_tensor(signal_x,kernel_x,loc_x)
    conv_y = build_meet_convolve_tensor(signal_y,kernel_y,loc_y)
    self.register_buffer('conv_x',conv_x)
    self.register_buffer('conv_y',conv_y)
    self.weights = nn.Parameter(torch.empty(kernel_x,kernel_y,in_features,out_features))
    self.bias = nn.Parameter(torch.empty(signal_x,signal_y,out_features))
    self.initialize_weights()

  def initialize_weights(self):
    nn.init.xavier_normal_(self.weights,gain=nn.init.calculate_gain('relu'))
    nn.init.normal_(self.bias)
    
  def forward(self, X):
    # X should be a (batchsize,signal_x,signal_y,in_features) tensor
    Y = torch.einsum("ixa,jyb,mijf,abzg->mxyg",self.conv_x,self.conv_y,X,self.weights)
    # Y is now (batchsize,signal_x,signal_y,out_features)
    return Y + self.bias #this should broadcast over batchsize
  
  def extra_repr(self):
    kernel_x, kernel_y, in_features, out_features = self.weights.shape
    return 'meet convolution layer with input_features={}, output_features={}, kernel_size={}'.format(
            in_features, out_features, (kernel_x,kernel_y))

class JoinConv2d(nn.Module):
  def __init__(self,signal_dim,kernel_dim,kernel_loc,in_features,out_features):
    super(JoinConv2d,self).__init__()
    (signal_x, signal_y) = signal_dim
    (kernel_x, kernel_y) = kernel_dim
    (loc_x, loc_y) = kernel_loc
    conv_x = build_join_convolve_tensor(signal_x,kernel_x,loc_x)
    conv_y = build_join_convolve_tensor(signal_y,kernel_y,loc_y)
    self.register_buffer('conv_x',conv_x)
    self.register_buffer('conv_y',conv_y)
    self.weights = nn.Parameter(torch.empty(kernel_x,kernel_y,in_features,out_features))
    self.bias = nn.Parameter(torch.empty(signal_x,signal_y,out_features))
    self.initialize_weights()

  def initialize_weights(self):
    nn.init.xavier_normal_(self.weights,gain=nn.init.calculate_gain('relu'))
    nn.init.normal_(self.bias)
    
  def forward(self, X):
    # X should be a (batchsize,signal_x,signal_y,in_features) tensor
    Y = torch.einsum("ixa,jyb,mijf,abzg->mxyg",self.conv_x,self.conv_y,X,self.weights)
    # Y is now (batchsize,signal_x,signal_y,out_features)
    return Y + self.bias #this should broadcast over batchsize
  
  def extra_repr(self):
    kernel_x, kernel_y, in_features, out_features = self.weights.shape
    return 'join convolution layer with input_features={}, output_features={}, kernel_size={}'.format(
            in_features, out_features, (kernel_x,kernel_y))

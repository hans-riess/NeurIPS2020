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
  # this is an inefficient implementation, but should be fast enough for the sizes we're doing.
  for i in range(signal_dim):
    for j in range(i,signal_dim):
      for k in range(kernel_dim):
        if i == max(j,k+kernel_loc):
          M[i,j,k] = 1
  return M


# to compute a convolution (f*g)_{xy}, we need to calculate the contraction M_{ixa} N_{jyb} f_{ij} g_{ab}
def lattice_convolution_2d(convolve_x,convolve_y,signal,kernel):
  return torch.einsum("ixa,jyb,ij,ab->xy",convolve_x,convolve_y,signal,kernel)

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
    self.bias = nn.Parameter(torch.empty(out_features,1,1))
    self.initialize_weights()

  def initialize_weights(self):
    nn.init.xavier_normal_(self.weights,gain=nn.init.calculate_gain('relu'))
    nn.init.normal_(self.bias)
    
  def forward(self, X):
    # X should be a (batchsize,in_features,signal_x,signal_y) tensor
    # Ok, this is gnarly. the contraction is M_ixa N_jyb X_mijf W_abfg.
    # M and N are the 1d shift tensors. So M_ixa X_mijf has spatial indices x, a, for X(x ^ a, - )
    # Similarly N_jyb X_mijf has spatial indices y, b, for X(-, y ^b).
    # So M_ixa N_yjb X_mijf represents X(x^a,y^b). Now we take the summation over a, b with the convolution kernel.
    # Finally, the summation over f takes the appropriate linear combination of all the convolutional kernels for this layer
    Y = torch.einsum("ixa,jyb,mfij,abfg->mgxy",self.conv_x,self.conv_y,X,self.weights)
    # Y is now (batchsize,signal_x,signal_y,out_features)
    return Y + self.bias #this should broadcast over everything
  
  def extra_repr(self):
    kernel_x, kernel_y, in_features, out_features = self.weights.shape
    return 'meet convolution layer with input_features={}, output_features={}, kernel_size={}'.format(in_features, out_features, (kernel_x,kernel_y))

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
    self.bias = nn.Parameter(torch.empty(out_features,1,1))
    self.initialize_weights()

  def initialize_weights(self):
    nn.init.xavier_normal_(self.weights,gain=nn.init.calculate_gain('relu'))
    nn.init.normal_(self.bias)
    
  def forward(self, X):
    # X should be a (batchsize,in_features,signal_x,signal_y) tensor
    Y = torch.einsum("ixa,jyb,mfij,abfg->mgxy",self.conv_x,self.conv_y,X,self.weights)
    # Y is now (batchsize,signal_x,signal_y,out_features)
    return Y + self.bias #this should broadcast over everything
  
  def extra_repr(self):
    kernel_x, kernel_y, in_features, out_features = self.weights.shape
    return 'join convolution layer with input_features={}, output_features={}, kernel_size={}'.format(in_features, out_features, (kernel_x,kernel_y))

class LatticeCNN(nn.Module):
  def __init__(self,signal_dim,kernel_dim,n_features):
    super(LatticeCNN,self).__init__()
    self.meet_conv = []
    self.join_conv = []
    for i in range(len(n_features)-1):
      self.meet_conv.append(MeetConv2d(signal_dim,kernel_dim,(signal_dim[0]-kernel_dim[0],signal_dim[1]-kernel_dim[1]),n_features[i],n_features[i+1]))
      self.join_conv.append(JoinConv2d(signal_dim,kernel_dim,(0,0),n_features[i],n_features[i+1]))

  def forward(self,x):
    for (mc,jc) in zip(self.meet_conv,self.join_conv):
      x = F.relu(mc(x) + jc(x))
    return x

class LatticeClassifier(nn.Module):
  def __init__(self,signal_dim,n_features,n_classes):
    super(LatticeClassifier,self).__init__()
    self.convolutions = LatticeCNN(signal_dim,(4,4),[n_features,16,16,8])
    self.fc1 = nn.Linear(8*signal_dim[0]*signal_dim[1],32)
    self.fc2 = nn.Linear(32,32)
    self.fc3 = nn.Linear(32,n_classes)
    self.sm = nn.Softmax(dim=0)

  def forward(self,x):
    batch_size = x.shape[0]
    x = self.convolutions(x)
    x = F.relu(self.fc1(torch.reshape(x,(batch_size,-1))))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    output = self.sm(x)
    return output

class ConvClassifier(nn.Module):
  def __init__(self,signal_dim,n_features,n_classes):
    super(ConvClassifier,self).__init__()
    self.convolutions = [nn.Conv2d(n_features,16,(4,4),1,padding=3),nn.Conv2d(16,16,(4,4),1,padding=3),nn.Conv2d(16,8,(4,4),1,padding=3)]
    self.fc1 = nn.Linear(8*signal_dim[0]*signal_dim[1],32)
    self.fc2 = nn.Linear(32,32)
    self.fc3 = nn.Linear(32,n_classes)
    self.sm = nn.Softmax(dim=0)
  def forward(self,x):
    batch_size = x.shape[0]
    for c in self.convolutions:
      x = F.relu(c(x))
    x = F.relu(self.fc1(torch.reshape(x,(batch_size,-1))))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    output = self.sm(x)
    return output
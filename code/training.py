import numpy as np
import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from rivet import parse_rivet_output
from neural import LatticeClassifier
from numpy.random import permutation
from torch.utils.data import DataLoader,random_split
import time

x_bins = 40
y_bins = 40
#data_path = "/home/ubuntu/code/data/ModelNet10/ModelNet10/"
data_path = 'C:/Users/hansr/OneDrive/Documents/Research/NeurIPS20/data/ModelNet10/ModelNet10/'

#binary classificaiton
categories = ['sofa','monitor']
n_features = 4
n_classes = len(categories)
feature_dim = (40,40)
label_dict = {category:index for index,category in enumerate(categories) }
reverse_label_dict = {index:category for index,category in enumerate(categories) }
#path = '/home/ubuntu/code/invariants/'
path = 'C:/Users/hansr/OneDrive/Documents/Research/NeurIPS20/code/invariants_binary/'
N = len(os.listdir(path))
files = os.listdir(path)
X = torch.zeros(N,n_features,x_bins,y_bins)
Y = torch.zeros(N)
for index,file_name in enumerate(files):
    X[index,:,:,:] = parse_rivet_output(path+file_name,x_bins,y_bins)
    v = label_dict[file_name.split('_')[1]]
    Y[index] = v
Y = Y.type(torch.LongTensor)
print(X.shape)
print(Y.shape)
data = [[X[index,:,:,:],Y[index]] for index in range(X.shape[0])]
training_data,testing_data = random_split(data,[len(data) - len(data)//10,len(data)//10])
trainloader = DataLoader(training_data,batch_size=5,shuffle=True)
testloader = DataLoader(testing_data,batch_size=5,shuffle=True)

#binary classification sofa vs. monitor
model = LatticeClassifier(feature_dim,n_features,n_classes)

start_time = time.time()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)
for epoch in range(2): 
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        a_loss = loss.item()
        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, a_loss))
print('Finished Training')

print("Training took %s seconds" % (time.time() - start_time))

PATH = './binary_class_3.pth'
torch.save(model.state_dict(), PATH)

total = 0
correct = 0
with torch.no_grad():
    for data in testloader:
        pointclouds, labels = data
        outputs = model(pointclouds)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (
    100.0 * correct / total))

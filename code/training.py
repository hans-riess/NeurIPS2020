import numpy as np
import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from rivet import parse_rivet_output
from neural import LatticeClassifier,ConvClassifier
from numpy.random import permutation
from torch.utils.data import DataLoader,random_split
import time
from matplotlib.pyplot import figure,xlabel,ylabel,plot,savefig
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

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

print('data has shape: '+ str(X.shape))
print('labels has shape: ' + str(Y.shape))

data = [[X[index,:,:,:],Y[index]] for index in range(X.shape[0])]
training_data,testing_data = random_split(data,[len(data) - len(data)//10,len(data)//10])
trainloader = DataLoader(training_data,batch_size=16,shuffle=True,pin_memory=True)
testloader = DataLoader(testing_data,batch_size=1,shuffle=True,pin_memory=True)

#binary classification sofa vs. monitor
n_trials = 5
n_epochs = 15

for trial in range(n_trials):
    start_time = time.time()
    model = LatticeClassifier(feature_dim,n_features,n_classes)
    model = model.to(device)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(n_epochs): 
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            a_loss = loss.item()
            print('[%d, %d, %5d] loss: %.3f' %
                    (trial + 1,epoch + 1, i + 1, a_loss))
        model_file = './trial_'+str(trial) + '_epoch_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), model_file)
    print('Finished Training')
    print("Training took %.3f seconds" % (time.time() - start_time))

accuracy = torch.zeros(n_epochs,n_trials)
with torch.no_grad():
    for trial in range(n_trials):
        for epoch in range(n_epochs):
            total = 0
            correct = 0
            model = LatticeClassifier(feature_dim,n_features,n_classes)
            model = model.to(device)
            model_file = './trial_'+str(trial) + '_epoch_'+str(epoch)+'.pth'
            model.load_state_dict(torch.load(model_file))
            for i,data in enumerate(testloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs,1)
                total+= labels.size(0)
                correct+= (predicted == labels).sum().item()
            accuracy[epoch,trial] = float(correct/total)
            print('Testing for trial %d, epoch %d complete!' %
                    (trial+1, epoch +1))

    accuracy = torch.mean(accuracy,dim = 0)
    torch.save(accuracy,'./testing_accuracy.pt')

#Create testing accuracy figure
plt.figure(figsize=(10,10))
plt.xlabel('Epochs')
plt.ylabel('Testing accuracy')
plt.plot(accuracy)
plt.savefig('./testing_accuracy')

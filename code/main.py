import os
from convert_point_cloud import off_to_pointcloud
from features import density_filtration
from rivet import write_rivet_input,rivet_find_invariants
import random

main_path = "/home/ubuntu/code/data/ModelNet10/ModelNet10/"
training_data = []
testing_data = []
for category in os.listdir(main_path):
    path_train = "/home/ubuntu/code/data/ModelNet10/ModelNet10/"+category+'/train/'
    for index,filename in enumerate(os.listdir(path_train)):
        if filename == ".DS_Store":
            continue
        path = path_train + filename
        print(filename)
        point_cloud = off_to_pointcloud(path)
        training_data.append((point_cloud,category,index))
    path_test = "/home/ubuntu/code/data/ModelNet10/ModelNet10/"+category+'/test/'
    for index,filename in enumerate(os.listdir(path_test)):
        if filename == ".DS_Store":
            continue
        path = path_test + filename
        print(filename)
        point_cloud = off_to_pointcloud(path)
        testing_data.append((point_cloud,category,index))

random.shuffle(training_data)
random.shuffle(testing_data)
k=100
x_bins = 100
y_bins = 100

for (point_cloud,label,index) in training_data:
    name = "training_point_cloud_" + str(label)+"-" + str(index)+".txt"
    write_rivet_input(point_cloud,density_filtration,k,name)
    for deg in range(2):
        output_name = 'training_invariants_deg_'+str(deg)+ "_"+ str(label)+"-"+str(index)+".txt"
        rivet_find_invariants(name,output_name,deg,x_bins,y_bins)

for (point_cloud,label,index) in testing_data:
    name = "testing_point_cloud_" + str(label)+"-"+str(index)+".txt"
    write_rivet_input(point_cloud,density_filtration,k,name)
    for deg in range(2):
        output_name = 'testing_invariants_deg_'+str(deg)+ "_"+ str(label)+"-"+str(index)+".txt"
        rivet_find_invariants(name,output_name,deg,x_bins,y_bins)

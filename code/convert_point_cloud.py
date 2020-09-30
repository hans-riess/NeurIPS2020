import numpy as np

def off_to_pointcloud(path):
    f = open(path,'r')
    f.readline()
    num_vertices = int(f.readline().split()[0])
    data_matrix = np.zeros([num_vertices,3])
    for i in range(num_vertices):
        x = np.array([float(t) for t in f.readline().split()])
        data_matrix[i,:]=x
    return data_matrix
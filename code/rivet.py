import os
import torch

def write_rivet_input(data_matrix,filtration_function,k,name):
    filtration_matrix = filtration_function(data_matrix,k)
    filtration_name = filtration_function.__name__
    newline = "\n"
    space = " "
    dim = str(filtration_matrix.shape[1]-1)
    f = open(name,"w")
    f.write("points"+newline)
    f.write(dim+newline)
    f.write("[-] "+filtration_name+newline)
    for row in filtration_matrix:
        f.write(str(row[0]) + space + str(row[1]) + space + str(row[2])+ newline)
    return None

def rivet_find_invariants(data_file_name,output_file_name, h_dim, x_bins,y_bins):
    cmd = "./rivet_console " + data_file_name + " --betti -H "+str(h_dim) + " -x "+str(x_bins)+" -y "+str(y_bins) + " > "+output_file_name
    os.system(cmd)
    return None

def str2tuple(string):
    return tuple([int(t) for t in string[1:-1].split(',')])

def parse_rivet_output(output_file_name, x_bins, y_bins):
    f = open(output_file_name,'r')
    lines = f.readlines()
    X = torch.zeros(4,x_bins,y_bins)
    d = 0
    for line in lines:
        if line.strip() == "x-grades":
            continue
        if line.strip() == "y-grades":
            continue
        if line.strip() == "Dimensions > 0:":
            #print(line)
            continue
        if line.strip() == "xi_0:":
            #print(line)
            d+=1
            continue
        if line.strip() == "xi_1:":
            #print(line)
            d+=1
            continue    
        if line.strip() == "xi_2:":
            #print(line)
            d+=1
            continue
        if line.strip() == '':
            continue
            
        if line.strip()[0] == '(' and line.strip()[-1] == ')':
            (x,y,f_xy) = str2tuple(line.strip())
            X[d,x,y] = float(f_xy)
    return X


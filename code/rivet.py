import os

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
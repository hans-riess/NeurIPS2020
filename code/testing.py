import numpy as np
from features import density_filtration
from rivet import rivet_find_invariants,write_rivet_input
k=10
a = np.random.rand(100,2)
print(a)
name = "test_point_cloud"
output_name = "test_invariants"
write_rivet_input(a,density_filtration,k,"test_point_cloud")
rivet_find_invariants(name,output_name,1,10,10)

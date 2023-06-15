import numpy as np 
import gzip
import matplotlib.pyplot as plt 


path = "monitoring/example/70"

U_path = path + "/U.gz"
p_path = path + "/U.gz"

with gzip.open(U_path, 'rb') as f:
    U = f.readlines()

def extract_vals(U):
    U = U[23:]
    i = 0 
    while list(str(U[i]).split("""b'""")[-1])[0] == '(':
        i += 1
    U_list = U[:i]
    new_U = []
    for U in U_list:
        U = str(U)
        U = U.split(' ')
        U[0] = float(U[0].split('(')[-1])
        U[1] = float(U[1])
        U[2] = float(U[2].split(')')[0])
        new_U.append(U)
    new_U = np.array(new_U)
    return new_U


import numpy as np 
import gzip
import matplotlib.pyplot as plt 



def extract_U_vals(U):
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

def extract_P_vals(P):
    P = P[23:]
    i = 0 
    while list(str(P[i]).split("""b'""")[-1])[0] != ')':
        i += 1
    P_list = P[:i]
    new_P = []
    for P in P_list:
        P = str(P)
        P = float(P.split("""b'""")[-1].split("""\\n""")[0])
        new_P.append(P)
    new_P = np.array(new_P)
    return new_P

def extract_data(path):
    U_path = path + "/U.gz"
    p_path = path + "/p.gz"

    with gzip.open(U_path, 'rb') as f:
        U = f.readlines()
    U = extract_U_vals(U)

    with gzip.open(p_path, 'rb') as f:
        P = f.readlines()
    P = extract_P_vals(P)
    return U,P


path = "emulation/simulations/36958981-e548-44ef-8528-d60c52cb4e94_r"
U,P = extract_data(path)
print(U.shape,P.shape)
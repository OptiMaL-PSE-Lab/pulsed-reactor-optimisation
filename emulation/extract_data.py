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

def extract_data(path):
    U_path = path + "/U.gz"
    p_path = path + "/U.gz"

    with gzip.open(U_path, 'rb') as f:
        U = f.readlines()
    U = extract_U_vals(U)
    # with gzip.open(p_path, 'rb') as f:
    #     P = f.readlines()
    # P = extract_vals(P)
    return U


path = "emulation/simulations/dd9abf2c-4e89-40c9-9a32-025ff5bf3641_r"
U = extract_data(path)
# print(U)
print(len(U))

# plt.figure()
# for i in range(3):
#     plt.plot(np.arange(len(U)),U[:,i])
# plt.show()


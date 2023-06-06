import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
import matplotlib.pyplot as plt 


data = read_json('symbolic_mf_data_generation/toy/data.json')['data']

x = [d['x']['x1'] for d in data]
z = [d['x']['z1'] for d in data]

plt.figure()
plt.scatter(x,z,c=np.arange(len(x)))
plt.show()
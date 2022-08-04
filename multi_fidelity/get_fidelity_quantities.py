import sys
import os 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mesh_generation.coil_basic import create_mesh
from uuid import uuid4
import numpy as np 
import pickle
from tqdm import tqdm 
from plot_fidelity_qualities import plot_fidelities
import shutil

MAR = np.zeros((10,5))
MNOM = np.zeros((10,5))
MNOA = np.zeros((10,5))
MS = np.zeros((10,5))

axial_fidelities = np.linspace(0,1,10)
radial_fidelities = [0,0.25,0.5,0.75,1]

for i in tqdm(range(len(axial_fidelities))):
        axial_fidelity = axial_fidelities[i]
        for j in range(len(radial_fidelities)):
                radial_fidelity = radial_fidelities[j]
                path = 'multi_fidelity/'+str(uuid4())
                create_mesh(0.012,0.0025,0.01,0.0753,None,[axial_fidelity,radial_fidelity],path,validation=True,build=True)

                with open(path+'/log.checkMesh', 'rt') as f:
                        data = f.readlines()
                for line in data:
                        if 'Max aspect ratio' in line:
                                MAR[i,j] = float(line.split('= ')[-1].split(' ')[0])
                        if 'Mesh non-orthogonality Max' in line:
                                MNOM[i,j] = float(line.split('Max: ')[-1].split(' ')[0])
                                MNOA[i,j] = float(line.split(': ')[-1].split(' ')[0])
                        if 'Max skewness' in line:
                                MS[i,j] = float(line.split('= ')[-1].split(' ')[0])
                plot_fidelities([MAR,MNOM,MNOA,MS])
                
                with open('multi_fidelity/fidelity_qualities.pickle', 'wb') as handle:
                        pickle.dump([MAR,MNOM,MNOA,MS], handle, protocol=pickle.HIGHEST_PROTOCOL)
                shutil.rmtree(path)
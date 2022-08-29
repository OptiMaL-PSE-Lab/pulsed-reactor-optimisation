import pandas as pd 
from utils import val_to_rtd 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import time 
import sys
import pickle
import os 
from utils import vel_calc,parse_conditions,run_cfd,calculate_N
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mesh_generation.coil_basic import create_mesh
import shutil 


def eval_cfd(a, f, re, pitch, coil_rad,inversion_loc,fid):
    tube_rad = 0.0025
    length = 0.0753
    identifier = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('Starting to mesh '+identifier)
    newcase = "outputs/initial_mf_points/" + identifier
    create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, fid,path=newcase,validation=False,build=True)
    print('Calculating Reynolds number')
    vel = vel_calc(re)
    print('Parsing conditions')
    parse_conditions(newcase, a, f, vel)
    print('Running CFD...')
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    #shutil.rmtree(newcase)
    return N,newcase

f1i = [0,0.25,0.5,0.75,1]
f2i = [0,0.25,0.5,0.75,1]

lb = np.array([0.001,2,10,0.0075,0.005,0])
ub = np.array([0.008,8,50,0.015,0.0125,1])
n_init = 8
dataset_x = []
for i in range(len(f1i)):

    init_points = np.random.uniform(0,1,(len(lb),n_init))
    init_points = np.array([[(init_points[i,j]) * (ub[i]-lb[i]) + lb[i] for i in range(len(lb))]for j in range(n_init)])

    f1 = f1i[i]
    f2 = f2i[i]

    
    for j in range(n_init):
        print(i,j)
        s = time.time()
        geom = init_points[j]
        N,newcase_path = eval_cfd(geom[0],geom[1],geom[2],geom[3],geom[4],geom[5],[f1,f2])
        e = time.time()
        x_dict = {'x':{}}
        x_dict['x']['a'] = geom[0]
        x_dict['x']['f'] = geom[1]
        x_dict['x']['re'] = geom[2]
        x_dict['x']['pitch'] = geom[3]
        x_dict['x']['coil_rad'] = geom[4]
        x_dict['x']['inversion_loc'] = geom[5]
        x_dict['x']['fid'] = f1
        x_dict['N'] = N
        x_dict['t'] = e-s
        x_dict['case'] = newcase_path
        dataset_x.append(x_dict)
        print(dataset_x)
        with open('outputs/initial_mf_points/dataset_x.pickle','wb') as file:
            pickle.dump(dataset_x,file)

    with open('outputs/initial_mf_points/dataset_x.pickle','wb') as file:
        pickle.dump(dataset_x,file)
with open('outputs/initial_mf_points/dataset_x.pickle','wb') as file:
    pickle.dump(dataset_x,file)
    

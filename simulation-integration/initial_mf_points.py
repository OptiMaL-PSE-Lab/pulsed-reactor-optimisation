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


def eval_cfd_validation(a, f, re, pitch, coil_rad,inversion_loc,fid):
    tube_rad = 0.0025
    length = 0.0753
    identifier = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    print('Starting to mesh '+identifier)
    newcase = "outputs/initial_mf_points/" + identifier
    create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, fid,path=newcase,validation=False,build=True)
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    return N,time,value,newcase

f1i = [0,0.25,0.5,0.75,1]
f2i = [0,0.25,0.5,0.75,1]

lb = np.array([0.001,2,10,0.0075,0.003,0])
ub = np.array([0.008,8,50,0.015,0.0125,1])
n_init = 5
t_list = np.zeros((len(f1i),n_init))
N_list = np.zeros((len(f1i),n_init))
points_list = []
for i in range(len(f1i)):

    init_points = np.random.uniform(0,1,(len(lb),n_init))
    init_points = np.array([[(init_points[i,j]) * (ub[i]-lb[i]) + lb[i] for i in range(len(lb))]for j in range(n_init)])

    f1 = f1i[i]
    f2 = f2i[i]

    for j in range(n_init):
        s = time.time()
        geom = init_points[j]
        N,time_list,value,path = eval_cfd_validation(geom[0],geom[1],geom[2],geom[3],geom[4],geom[5],[f1,f2])
        
        e = time.time()
        t_list[i,j] = e-s
        N_list[i,j] = N

    points_list.append(init_points)
    with open('outputs/validation/initial_mf_points.pickle','wb') as file:
        pickle.dump([t_list,N_list,points_list],file)
with open('outputs/validation/initial_mf_points.pickle','wb') as file:
    pickle.dump([t_list,N_list,points_list],file)
    
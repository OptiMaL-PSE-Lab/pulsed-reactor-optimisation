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

HPC = True 

def eval_cfd_validation(a, f, re, coil_rad, pitch,inversion_loc,fid):
    tube_rad = 0.0025
    length = 0.0753
    identifier = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    print('Starting to mesh '+identifier)
    newcase = "outputs/validation/" + identifier
    create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, fid,path=newcase,validation=True,build=True)
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    return N,time,value,newcase

# fig,ax = plt.subplots(1,2,figsize=(8,4))
# fig.subplots_adjust(wspace=0.2,bottom=0.15,right=0.975,left=0.075)
# for i in range(2):
#     ax[i].set_xlabel(r'$\theta$')
#     ax[i].set_ylabel(r'$E({\theta})$')
#     ax[i].set_ylim(-0.1,2.1)
#     ax[i].set_xlim(0,2)

# ax[0].scatter(t1,e1,c='k',s=10,marker='x',lw=0.75,zorder=10)
# ax[1].scatter(t2,e2,c='k',s=10,marker='x',lw=0.75,label='Experiment',zorder=10)
# ax[0].set_title('Experiment 1')
# ax[1].set_title('Experiment 2')

f1i = [0,0.25,0.5,0.75,1]
f2i = [0,0.25,0.5,0.75,1]
#color = iter(cm.coolwarm(np.linspace(0, 1, 5)))

lb = np.array([0.001,2,10,0.003,0.0075,0])
ub = np.array([0.008,8,50,0.0125,0.015,1])
n_init = 5
t_list = np.zeros((len(f1i),n_init))
N_list = np.zeros((len(f1i),n_init))
points_list = []
for i in range(len(f1i)):

    init_points = np.random.uniform(0,1,(len(lb),n_init))
    init_points = np.array([[(init_points[i,j] + lb[i]) *(ub[i]-lb[i]) for i in range(len(lb))]for j in range(n_init)])

    f1 = f1i[i]
    f2 = f2i[i]

    #c = next(color)
    for j in range(n_init):
        s = time.time()
        geom = init_points[j]
        N,time_list,value,path = eval_cfd_validation(geom[0],geom[1],geom[2],geom[3],geom[4],geom[5],[f1,f2])
        
        e = time.time()
        t_list[i,j] = e-s
        N_list[i,j] = N
        #shutil.rmtree(path)
    points_list.append(init_points)
    with open('outputs/validation/initial_points.pickle','wb') as file:
        pickle.dump([t_list,N_list,points_list],file)
with open('outputs/validation/initial_points.pickle','wb') as file:
    pickle.dump([t_list,N_list,points_list],file)
    
#     theta,etheta = val_to_rtd(time,value,path)

#     # df = pd.read_csv('simulation-integration/output_validation/e_1_fid_'+str(f)+'.csv')
#     # theta = df['theta'].values
#     # etheta = df['etheta'].values
#     #ax[0].plot(theta,etheta,c=c)

#     res1 = pd.DataFrame({'time':time,'concentration':value})
#     res2 = pd.DataFrame({'theta':theta,'etheta':etheta})
#     res = pd.concat([res1, res2], axis=1) 
#     res.to_csv('outputs/validation/e_1_fid_'+str(f1)+'_'+str(f2)+'.csv')
#     #fig.savefig('simulation-integration/output_validation/experimental_validation.pdf')

#     N,time,value,path = eval_cfd_validation(0.004,5,50,0.012,0.01,0.0025,0.0753,[f1,f2])
#     theta,etheta = val_to_rtd(time,value,path)

#     # df = pd.read_csv('simulation-integration/output_validation/e_2_fid_'+str(f)+'.csv')
#     # theta = df['theta'].values
#     # etheta = df['etheta'].values
#     #ax[1].plot(theta,etheta,c=c,label='Fidelity: '+str(i+1))
#     #ax[0].grid(alpha=0.5)
#     #ax[1].grid(alpha=0.5)
#     res1 = pd.DataFrame({'time':time,'concentration':value})
#     res2 = pd.DataFrame({'theta':theta,'etheta':etheta})
#     res = pd.concat([res1, res2], axis=1) 
#     res.to_csv('outputs/validation/e_2_fid_'+str(f1)+'_'+str(f2)+'.csv')
#     #fig.savefig('simulation-integration/output_validation/experimental_validation.pdf')

# #ax[1].legend(frameon=False)
# f#ig.savefig('simulation-integration/output_validation/experimental_validation.pdf')


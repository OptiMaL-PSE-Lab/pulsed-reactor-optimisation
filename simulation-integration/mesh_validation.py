import pandas as pd 
from utils import val_to_rtd 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import time 

def eval_cfd_validation(a, f, re, coil_rad, pitch,tube_rad,length,fid):
    inversion_loc = None
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

f1i = [0.25,0.75,1,1.25]
f2i = [0.25,0.75,1,1.25]
#color = iter(cm.coolwarm(np.linspace(0, 1, 5)))

for i in range(len(f1i)):

    f1 = f1i[i]
    f2 = f2i[i]

    #c = next(color)

    N,time,value,path = eval_cfd_validation(0.002,5,50,0.012,0.01,0.0025,0.0753,[f1,f2])


    
    theta,etheta = val_to_rtd(time,value,path)

    # df = pd.read_csv('simulation-integration/output_validation/e_1_fid_'+str(f)+'.csv')
    # theta = df['theta'].values
    # etheta = df['etheta'].values
    #ax[0].plot(theta,etheta,c=c)

    res1 = pd.DataFrame({'time':time,'concentration':value})
    res2 = pd.DataFrame({'theta':theta,'etheta':etheta})
    res = pd.concat([res1, res2], axis=1) 
    res.to_csv('outputs/validation/e_1_fid_'+str(f1)+'_'+str(f2)+'.csv')
    #fig.savefig('simulation-integration/output_validation/experimental_validation.pdf')

    N,time,value,path = eval_cfd_validation(0.004,5,50,0.012,0.01,0.0025,0.0753,[f1,f2])
    theta,etheta = val_to_rtd(time,value,path)

    # df = pd.read_csv('simulation-integration/output_validation/e_2_fid_'+str(f)+'.csv')
    # theta = df['theta'].values
    # etheta = df['etheta'].values
    #ax[1].plot(theta,etheta,c=c,label='Fidelity: '+str(i+1))
    #ax[0].grid(alpha=0.5)
    #ax[1].grid(alpha=0.5)
    res1 = pd.DataFrame({'time':time,'concentration':value})
    res2 = pd.DataFrame({'theta':theta,'etheta':etheta})
    res = pd.concat([res1, res2], axis=1) 
    res.to_csv('outputs/validation/e_2_fid_'+str(f1)+'_'+str(f2)+'.csv')
    #fig.savefig('simulation-integration/output_validation/experimental_validation.pdf')

#ax[1].legend(frameon=False)
f#ig.savefig('simulation-integration/output_validation/experimental_validation.pdf')


import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cylindrical import create_mesh


coils = 4  # number of coils
h = coils * 0.0075  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 8 # points per coil to use

data = {}
nominal_data = {}


z_vals = np.linspace(0,h,n)
theta_vals = np.linspace(0, N, n)
rho_vals = [0.0035 for i in range(n)]
tube_rad_vals = [0.0025 for i in range(n)]
for i in range(n):
    nominal_data['z_'+str(i)] = z_vals[i]
    nominal_data['theta_'+str(i)] = theta_vals[i]
    nominal_data["tube_rad_"+str(i)] = tube_rad_vals[i]
    nominal_data["rho_"+str(i)] = rho_vals[i]

a = 0.001
f = 2
re = 50

z_bounds = {}
z_bounds['fid_axial'] = [9.51,20.49]
z_bounds['fid_radial'] = [0.51,6.49]

x_bounds = {}
for i in range(n):
    x_bounds['rho_'+str(i)] = [0,0.01]
    x_bounds['z_'+str(i)] = [-0.001,0.001]

data_path = "parameterisation_study/data.json"

def eval_cfd(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ID = 'test'
    case = data_path.split('/')[0]+'/simulations/'+ID
    create_mesh(
        x,
        case,
        n,
        nominal_data,
    )
    parse_conditions_given(case, a,f,re)
    times, values = run_cfd(case)
    N = calculate_N(values, times, case)
    for i in range(48):
        shutil.rmtree(case + "/processor" + str(i))
    # shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N, "cost": end - start, "id": ID}

mfbo(eval_cfd,data_path,x_bounds,z_bounds,gamma=1.5,beta=2.5,p_c=2,sample_initial=25,plot_only=False,debug=False)
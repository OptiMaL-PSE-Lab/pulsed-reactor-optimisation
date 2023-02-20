import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cylindrical import create_mesh


coils = 3  # number of coils
h = 20  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 8 # points per coil to use

data = {}
nominal_data = {}


z_vals = np.linspace(0,h,n)
theta_vals = np.linspace(0, N, n)
rho_vals = [4.0 for i in range(n)]
tube_rad_vals = [1.5 for i in range(n)]
for i in range(n):
    nominal_data['z_'+str(i)] = z_vals[i]
    nominal_data['theta_'+str(i)] = theta_vals[i]
    nominal_data["tube_rad_"+str(i)] = tube_rad_vals[i]
    nominal_data["rho_"+str(i)] = rho_vals[i]

z_bounds = {}
z_bounds['fid_axial'] = [3.51,16.49]
z_bounds['fid_radial'] = [0.51,6.49]

x_bounds = {}
for i in range(n):
    x_bounds['rho_'+str(i)] = [-3,3]
    x_bounds['z_'+str(i)] = [-(h/(2*n)),h/(2*n)]

data_path = "parameterisation_study/data.json"

def eval_cfd(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = data_path.split('/')[0]+'/simulations/'+ID
    create_mesh(
        x,
        case,
        n,
        nominal_data,
    )
    parse_conditions(case, x)
    times, values = run_cfd(case)
    N = calculate_N(values, times, case)
    for i in range(48):
        shutil.rmtree(case + "/processor" + str(i))
    # shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N, "cost": end - start, "id": ID}

mfbo(eval_cfd,data_path,x_bounds,z_bounds,gamma=1.5,beta=2.5,p_c=2,sample_initial=True,plot_only=False,debug=False)
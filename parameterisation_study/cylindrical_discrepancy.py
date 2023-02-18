import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cylindrical import create_mesh



coils = 3  # number of coils
coil_rad_max = 10  # max coil radius
h = 20  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 8 # points per coil to use
# example data for cylindrical coordinates (normally optimized for)
# as close as possible to the paper
data = {}

z_vals = np.linspace(0,h,n)
theta_vals = np.linspace(0, N, n)
rho_vals = [4.0 for i in range(n)]
tube_rad_vals = [1.5 for i in range(n)]
for i in range(n):
    data['z_'+str(i)] = z_vals[i]
    data['theta_'+str(i)] = theta_vals[i]
    data["tube_rad_"+str(i)] = tube_rad_vals[i]
    data["rho_"+str(i)] = rho_vals[i]

data["fid_radial"] = 16 # between 4 and 16

# rho_desc = np.random.uniform(-3,+3, n)
# z_desc = np.random.uniform(-(h/(2*n)),h/(2*n),n)
# for i in range(n):
#     data['rho_'+str(i)] = rho_desc[i]
#     data['z_'+str(i)] = z_desc[i]


# create mesh from cylindrical coordinates

data_path = "parameterisation_study/data_cd.json"
ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
case = data_path.split('/')[0]+'/simulations/'+ID

case = 'parameterisation_study/simulations/test'

create_mesh(data,case,n)


# x_bounds = {}
# x_bounds["a"] = [0.001, 0.008]
# x_bounds["f"] = [2, 8]
# x_bounds["re"] = [10, 50]
# x_bounds["pitch"] = [0.0075, 0.02]
# x_bounds["coil_rad"] = [0.0035, 0.0125]
# x_bounds["inversion_loc"] = [0, 1]

# z_bounds = {}
# z_bounds["fid_axial"] = [20.001, 49.99]
# z_bounds["fid_radial"] = [1.001, 5.99]

# data_path = "parameterisation_study/data_cd.json"

# def eval_cfd(x: dict):
#     start = time.time()
#     ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     case = data_path.split('/')[0]+'/simulations/'+ID
#     create_mesh(
#         x,
#         length=0.0753,
#         tube_rad=0.0025,
#         path=case,
#     )
#     parse_conditions(case, x)
#     times, values = run_cfd(case)
#     N = calculate_N(values, times, case)
#     for i in range(16):
#         shutil.rmtree(case + "/processor" + str(i))
#     # shutil.rmtree(newcase)
#     end = time.time()
#     return {"obj": N, "cost": end - start, "id": ID}

# mfbo(eval_cfd,data_path,x_bounds,z_bounds,gamma=1.5,beta=2.5,p_c=2,sample_initial=25,plot_only=False)

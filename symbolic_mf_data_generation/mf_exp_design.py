import sys
import os
from utils_ed import * 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from symbolic_mf_data_generation.main_ed import mfed
from mesh_generation.coil_basic import create_mesh
from jax import grad, jit, value_and_grad
import jax.numpy as jnp
from utils_plotting import *
from utils_gp import *


x_bounds = {}
x_bounds["a"] = [0.001, 0.008]
x_bounds["f"] = [2, 8]
x_bounds["re"] = [10, 50]
x_bounds["pitch"] = [0.0075, 0.02]
x_bounds["coil_rad"] = [0.0035, 0.0125]
x_bounds["inversion_loc"] = [0, 1]

z_bounds = {}
z_bounds["fid_axial"] = [19.51, 50.49]
z_bounds["fid_radial"] = [1.51, 5.49]


try:
    data_path = str(sys.argv[1])
    gamma = float(sys.argv[2])
    cpus = int(sys.argv[3])
except IndexError:
    data_path = 'symbolic_mf_data_generation/exp_design/data.json'
    gamma = 1.5 
    cpus = 2

cpu_vals = derive_cpu_split(cpus)

shutil.copy("mesh_generation/mesh/system/default_decomposeParDict","mesh_generation/mesh/system/decomposeParDict")
replaceAll("mesh_generation/mesh/system/decomposeParDict","numberOfSubdomains 48;","numberOfSubdomains "+str(cpus)+";")
replaceAll("mesh_generation/mesh/system/decomposeParDict","    n               (4 4 3);","    n               ("+str(cpu_vals[0])+" "+str(cpu_vals[1])+" "+str(cpu_vals[2])+");")


try:
    print('Building simulation folder')
    os.mkdir(data_path.split("data.json")[0] + "simulations/")
except FileExistsError:
    print('Simulation folder already exists')


def eval_cfd(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = data_path.split('data.json')[0]+'simulations/'+ID
    create_mesh(
        x,
        length=0.0753,
        tube_rad=0.0025,
        path=case,
    )
    parse_conditions(case, x)
    times, values = run_cfd(case)
    N = calculate_N(values, times, case)
    for i in range(cpus):
        shutil.rmtree(case + "/processor" + str(i))
    # shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N, "cost": end - start, "id": ID}




# gen_data(eval_cfd, data_path, x_bounds, {"fid_axial": fid_ax,"fid_radial":fid_rad}, n_s)


# mfed(eval_cfd, data_path, x_bounds, z_bounds,120*48*48,gamma=gamma,sample_initial=8,int_fidelities=[True,True])


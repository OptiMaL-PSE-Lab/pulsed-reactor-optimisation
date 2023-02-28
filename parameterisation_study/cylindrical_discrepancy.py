import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cylindrical import create_mesh


coils = 4  # number of coils
h = coils * 0.0075  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 8  # points per coil to use

data = {}
nominal_data = {}


z_vals = np.linspace(0, h, n)
theta_vals = np.linspace(0+np.pi/2, N+np.pi/2, n)
rho_vals = [0.0035 for i in range(n)]
tube_rad_vals = [0.0025 for i in range(n)]
for i in range(n):
    nominal_data["z_" + str(i)] = z_vals[i]
    nominal_data["theta_" + str(i)] = theta_vals[i]
    nominal_data["tube_rad_" + str(i)] = tube_rad_vals[i]
    nominal_data["rho_" + str(i)] = rho_vals[i]



z_bounds = {}
z_bounds["fid_axial"] = [9.51, 25.49]
z_bounds["fid_radial"] = [0.51, 5.49]

x_bounds = {}
for i in range(n):
    x_bounds["rho_" + str(i)] = [0, 0.01]
    x_bounds["z_" + str(i)] = [-0.001, 0.001]


data_path = str(sys.argv[1])
gamma = float(sys.argv[2])
beta = float(sys.argv[3])
p_c = float(sys.argv[4])

try:
    print('Building simulation folder')
    os.mkdir(data_path.split("data.json")[0] + "simulations/")
except FileExistsError:
    print('Simulation folder already exists')


def eval_cfd(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = data_path.split("data.json")[0] + "simulations/" + ID
    create_mesh(
        x,
        case,
        n,
        nominal_data,
    )
    a = 0.001
    f = 2
    re = 50
    parse_conditions_given(case, a, f, re)
    times, values = run_cfd(case)
    N = calculate_N(values, times, case)
    for i in range(512):
        try:
            shutil.rmtree(case + "/processor" + str(i))
        except:
            print('no folder here')
    # shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N, "cost": end - start, "id": ID}


mfbo(eval_cfd, data_path, x_bounds, z_bounds,64*60*60,gamma=gamma, beta=beta, p_c=p_c,sample_initial=16,int_fidelities=True)

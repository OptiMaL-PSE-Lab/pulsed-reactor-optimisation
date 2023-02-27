import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cross_section import create_mesh


n = 6
n_cross_section = 6
coil_data = {"start_rad":0.00275,"radius_center":0.00125,"length":0.0753,"pitch":0.01,"coil_rad":0.0075,"fid_axial":8,"fid_radial":5}

z_bounds = {}
z_bounds["fid_axial"] = [5, 15]
z_bounds["fid_radial"] = [1, 6]

x_bounds = {}
for i in range(n):
    for j in range(n):
        x_bounds["r_" + str(i)+'_'+str(j)] = [0.0015, 0.0035]

data_path = str(sys.argv[1])
gamma = float(sys.argv[2])
beta = float(sys.argv[3])
p_c = float(sys.argv[4])

# cross_section = [np.array([0.0025 for i in range(n_cross_section)]) for i in range(n)]

try:
    print('Building simulation folder')
    os.mkdir(data_path.split("data.json")[0] + "simulations/")
except FileExistsError:
    print('Simulation folder already exists')


def eval_cfd(x: dict):

    coil_data['fid_radial'] = x['fid_radial']
    coil_data['fid_axial'] = x['fid_axial']

    x_list = []
    for i in range(n):
        x_add = []
        for j in range(n_cross_section):
            x_add.append(x['r_' + str(i) + '_' + str(j)])

        x_list.append(np.array(x_add))

    a = 0.001
    f = 2
    re = 50
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = data_path.split("data.json")[0] + "simulations/" + ID
    create_mesh(x_list,coil_data,case,debug=False)

    parse_conditions_given(case, a, f, re)
    times, values = run_cfd(case)
    N = calculate_N(values, times, case)
    for i in range(512):
        shutil.rmtree(case + "/processor" + str(i))
    #shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N, "cost": end - start, "id": ID}


mfbo(eval_cfd, data_path, x_bounds, z_bounds,64*48*48,gamma=gamma, beta=beta, p_c=p_c,sample_initial=64,int_fidelities=True)
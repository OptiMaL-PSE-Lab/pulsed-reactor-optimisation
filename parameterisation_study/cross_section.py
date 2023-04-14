import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cross_section import create_mesh


n_circ = 6
n_cross_section = 6
coils = 2
length = np.pi * 2 * 0.010391 * coils
coil_data = {"start_rad":0.0025,"radius_center":0.00125,"length":length,"a": 0.0009999999310821295, "f": 2.0, "re": 50.0, "pitch": 0.010391080752015114, "coil_rad": 0.012500000186264515, "inversion_loc": 0.6596429944038391}
z_bounds = {}
z_bounds["fid_axial"] = [15.55, 40.45]
z_bounds["fid_radial"] = [1.55, 4.45]


x_bounds = {}
for i in range(n_circ):
    for j in range(n_cross_section):
        x_bounds["r_" + str(i)+'_'+str(j)] = [0.002, 0.004]

try:
    data_path = str(sys.argv[1])
    gamma = float(sys.argv[2])
    beta = float(sys.argv[3])
    p_c = float(sys.argv[4])
    cpus = int(sys.argv[5])
except IndexError:
    data_path = 'parameterisation_study/cross_section/data.json'
    gamma = 2.5 
    beta = 1.5 
    p_c = 2
    cpus = 1

cpu_vals = derive_cpu_split(cpus)

shutil.copy("mesh_generation/mesh/system/default_decomposeParDict","mesh_generation/mesh/system/decomposeParDict")
replaceAll("mesh_generation/mesh/system/decomposeParDict","numberOfSubdomains 48;","numberOfSubdomains "+str(int(cpus))+";")
replaceAll("mesh_generation/mesh/system/decomposeParDict","    n               (4 4 3);","    n               ("+str(cpu_vals[0])+" "+str(cpu_vals[1])+" "+str(cpu_vals[2])+");")


# cross_section = [np.array([0.0025 for i in range(n_cross_section)]) for i in range(n)]

try:
    print('Building simulation folder')
    os.mkdir(data_path.split("data.json")[0])
except FileExistsError:
    print('Simulation folder already exists')

try:
    print('Building simulation folder')
    os.mkdir(data_path.split("data.json")[0] + "simulations/")
except FileExistsError:
    print('Simulation folder already exists')


def eval_cfd(x: dict):

    coil_data['fid_radial'] = x['fid_radial']
    coil_data['fid_axial'] = x['fid_axial']

    x_list = []
    for i in range(n_circ):
        x_add = []
        for j in range(n_cross_section):
            x_add.append(x['r_' + str(i) + '_' + str(j)])

        x_list.append(np.array(x_add))

    a = 0
    f = 0
    re = 50
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = data_path.split("data.json")[0] + "simulations/" + ID

    create_mesh(x_list,coil_data.copy(),case,debug=False)

    parse_conditions_given(case, a, f, re)
    times, values = run_cfd(case)
    N,penalty = calculate_N_clean(values, times, case)
    for i in range(cpus):
        shutil.rmtree(case + "/processor" + str(i))
    #shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N-penalty, "TIS": N, "penalty": penalty, "cost": end - start, "id": ID}


mfbo(eval_cfd, data_path, x_bounds, z_bounds,72*60*60,gamma=gamma, beta=beta, p_c=p_c,gp_ms=4,opt_ms=8,sample_initial=False,int_fidelities=[True,True])

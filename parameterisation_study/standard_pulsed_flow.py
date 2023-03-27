import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cylindrical import create_mesh



z_bounds = {}
z_bounds["fid_axial"] = [15.55, 40.55]
z_bounds["fid_radial"] = [2.55, 6.45]

x_bounds = {}
x_bounds["a"] = [0.001, 0.008]
x_bounds["f"] = [2, 8]
x_bounds["re"] = [10, 50]
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

    coils = 3  # number of coils
    h = coils * 0.0103  # max height
    N = 2 * np.pi * coils  # angular turns (radians)
    n = 6  # points to use

    data = {}
    nominal_data = {}

    z_vals = np.linspace(0, h, n)
    theta_vals = np.flip(np.linspace(0+np.pi/2, N+np.pi/2, n))
    rho_vals = [0.0125 for i in range(n)]
    tube_rad_vals = [0.0025 for i in range(n)]
    for i in range(n):
        nominal_data["z_" + str(i)] = z_vals[i]
        nominal_data["theta_" + str(i)] = theta_vals[i]
        nominal_data["tube_rad_" + str(i)] = tube_rad_vals[i]
        nominal_data["rho_" + str(i)] = rho_vals[i]
        x['rho_' + str(i)] = 0
        x['z_' + str(i)] = 0

    a = x['a']
    f = x['f']
    re = x['re']
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = data_path.split("data.json")[0] + "simulations/" + ID

    # create_mesh(x_list,coil_data.copy(),case,debug=False)
    create_mesh(x,case,n,nominal_data)

    parse_conditions_given(case, a, f, re)
    times, values = run_cfd(case)
    N,penalty = calculate_N(values, times, case)
    for i in range(cpus):
        shutil.rmtree(case + "/processor" + str(i))
    #shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N-penalty, "TIS": N, "penalty": penalty, "cost": end - start, "id": ID}


mfbo(eval_cfd, data_path, x_bounds, z_bounds,72*60*60,gamma=gamma, beta=beta, p_c=p_c,sample_initial=16,int_fidelities=[True,True])

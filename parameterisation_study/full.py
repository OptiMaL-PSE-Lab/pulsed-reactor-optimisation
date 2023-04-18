import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_both import create_mesh



z_bounds = {}
z_bounds["fid_axial"] = [15.55, 40.45]
z_bounds["fid_radial"] = [1.55, 4.45]

n_circ = 6
n_cross_section = 6
n = 6
x_bounds = {}
for i in range(n_circ):
    for j in range(n_cross_section):
        x_bounds["r_" + str(i)+'_'+str(j)] = [0.002, 0.004]

x_bounds['z_1'] = [-0.001,0.001]
for i in range(2,n):
	x_bounds['z_'+str(i)] = [-0.001,0.001]
	x_bounds['rho_'+str(i)] = [-0.0035,0.0035]

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
    data_path = 'parameterisation_study/full/data.json'
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

    # data['fid_radial'] = x['fid_radial']
    # data['fid_axial'] = x['fid_axial']

    data = {}
    data['fid_radial'] = x['fid_radial']
    data['fid_axial'] = x['fid_axial']
    nominal_data = {}

    n_circ = 6
    n_cross_section = 6
    coils = 2
    h = coils * 0.010391  # max height
    N = 2*np.pi *coils  # angular turns (radians)
    n = 6

    data['rho_0'] = 0
    data['z_0'] = 0
    data['rho_1'] = 0
    data['z_1'] = x['z_1'] * 2
    for i in range(2,n):
            data['z_'+str(i)] = x['z_'+str(i)] * 2
            data['rho_'+str(i)] = x['rho_'+str(i)] * 2

    z_vals = np.linspace(0, h, n)
    theta_vals = np.flip(np.linspace(0+np.pi/2, N+np.pi/2, n))
    rho_vals = [0.0125 for i in range(n)]
    tube_rad_vals = [0.0025 for i in range(n)]
    for i in range(n):
            nominal_data["z_" + str(i)] = z_vals[i]
            nominal_data["theta_" + str(i)] = theta_vals[i]
            nominal_data["tube_rad_" + str(i)] = tube_rad_vals[i]
            nominal_data["rho_" + str(i)] = rho_vals[i]

    x_list = []
    for i in range(n_circ):
        x_add = []
        for j in range(n_cross_section):
            x_add.append(x['r_' + str(i) + '_' + str(j)])

        x_list.append(np.array(x_add))

    a = x['a']
    f = x['f']
    re = x['re']
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = data_path.split("data.json")[0] + "simulations/" + ID
    # case = "parameterisation_study/both/simulations/" + str('test')

    create_mesh(data,x_list,case,n,nominal_data)

    parse_conditions_given(case, a, f, re)
    times, values = run_cfd(case)
    N,penalty = calculate_N(values, times, case)
    for i in range(cpus):
        shutil.rmtree(case + "/processor" + str(i))
    #shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N-penalty, "TIS": N, "penalty": penalty, "cost": end - start, "id": ID}


mfbo(eval_cfd, data_path, x_bounds, z_bounds,168*60*60,gamma=gamma, beta=beta, p_c=p_c,gp_ms=4,opt_ms=8,sample_initial=32,int_fidelities=[True,True])

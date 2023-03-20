import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cross_section_fid_study import create_mesh


try:
    gamma = float(sys.argv[1])
    beta = float(sys.argv[2])
    p_c = float(sys.argv[3])
    cpus = int(sys.argv[4])
    testing = str(sys.argv[5])
except IndexError:
    gamma = 2.5 
    beta = 1.5 
    p_c = 2
    cpus = 1
    testing = 'axial_radial_deltaT_maxCo'

data_path = 'what_is_a_fidelity/'+testing+'/data.json'


n_circ = 4
n_cross_section = 6
coils = 2
length = np.pi * 2 * 0.0125 * coils
coil_data = {"start_rad":0.0025,"radius_center":0.00125,"length":length,"a": 0.0009999999310821295, "f": 2.0, "re": 50.0, "pitch": 0.010391080752015114, "coil_rad": 0.012500000186264515, "inversion_loc": 0.6596429944038391, "fid_axial": 50, "fid_radial": 5}

z_bounds = {}

nominal_z = {'fid_axial':20.55,'fid_radial':4,'fid_deltaT':0.00001,'fid_maxCo':5}
fids = {'axial':[10.55,30.55],'radial':[2.55,6.45],'deltaT':[0.0001,0.0000001],'maxCo':[2,10]}
z_int = []
for f in fids:
    if f in testing:
        if f == 'axial' or f == 'radial':
            z_int.append(True)
        else:
            z_int.append(False)

        z_bounds["fid_"+f] = fids[f]


x_bounds = {}
for i in range(n_circ):
    for j in range(n_cross_section):
        x_bounds["r_" + str(i)+'_'+str(j)] = [0.002, 0.004]


cpu_vals = derive_cpu_split(cpus)
if cpus == 1:
    cpu_vals = [1,1,1]

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

    try:
        coil_data['fid_radial'] = x['fid_radial']
    except KeyError:
        coil_data['fid_radial'] = nominal_z['fid_radial']

    try:
        coil_data['fid_axial'] = x['fid_axial']
    except KeyError:
        coil_data['fid_axial'] = nominal_z['fid_axial']

    try:
        coil_data['fid_deltaT'] = x['fid_deltaT']
    except KeyError:
        coil_data['fid_deltaT'] = nominal_z['fid_deltaT']

    try:
        coil_data['fid_maxCo'] = x['fid_maxCo']
    except KeyError:
        coil_data['fid_maxCo'] = nominal_z['fid_maxCo']

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


mfbo(eval_cfd, data_path, x_bounds, z_bounds,72*60*60,gamma=gamma, beta=beta, p_c=p_c,sample_initial=32,int_fidelities=z_int)

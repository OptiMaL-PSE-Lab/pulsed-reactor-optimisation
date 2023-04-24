import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cylindrical import create_mesh as create_mesh_cylinder
from mesh_generation.coil_basic import create_mesh as create_mesh_basic
from mesh_generation.coil_cross_section import create_mesh as create_mesh_cross_section

try: 
    flag = sys.argv[2]
except:
    flag = 'standard'
coils = 2  # number of coils
coils = 2  # number of coils
pitch = 0.010391
rad = 0.0125
h = coils * pitch  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 6  # points to use

n_circ = 6
n_cross_section = 6
length = np.pi * 2 * rad * coils
coil_data = {"start_rad":0.0025,"radius_center":0.00125,"length":length,"pitch": pitch, "coil_rad": rad}

def eval_cfd_cross(x: dict):

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
    case = "parameterisation_study/cross_standard_simulation"

    create_mesh_cross_section(x_list,coil_data.copy(),case,debug=False)
    n = 8  # points to use

    parse_conditions_given(case, a, f, re)
    times, values = run_cfd(case)
    N,penalty = calculate_N_clean(values, times, case)
    for i in range(cpus):
        shutil.rmtree(case + "/processor" + str(i))
    #shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N-penalty, "TIS": N, "penalty": penalty, "cost": end - start, "id": ID}

def eval_cfd_basic(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = "parameterisation_study/basic_standard_simulation_smaller"
    create_mesh_basic(
        x,
        length=pitch*2*np.pi*coils,
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

def eval_cfd_cylinder(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = "parameterisation_study/cylinder_standard_simulation_smaller"
    create_mesh_cylinder(
        x,
        case,
        n,
        nominal_data.copy(),
    )
    a = 0
    f = 0
    re = 50
    parse_conditions_given(case, a, f, re)
    times, values = run_cfd(case)
    N,penalty = calculate_N_clean(values, times, case)
    for i in range(cpus):
        shutil.rmtree(case + "/processor" + str(i))
    # shutil.rmtree(newcase)
    end = time.time()
    #return {"obj": 0, "cost": end - start, "id": ID}
    return {"obj": N-penalty, "TIS": N, "penalty": penalty, "cost": end - start, "id": ID}



if flag == 'cylinder':
    data = {}
    nominal_data = {}

    data['rho_0'] = 0
    data['z_0'] = 0
    data['z_1'] = 0
    data['rho_1'] = 0
    for i in range(2,n):
        data['z_'+str(i)] = 0
        data['rho_'+str(i)] = 0
    z_vals = np.linspace(0, h/2, n)
    theta_vals = np.flip(np.linspace(0+np.pi/2, N+np.pi/2, n))
    rho_vals = [rad/2 for i in range(n)]
    tube_rad_vals = [0.0025/2 for i in range(n)]
    for i in range(n):
        nominal_data["z_" + str(i)] = z_vals[i]
        nominal_data["theta_" + str(i)] = theta_vals[i]
        nominal_data["tube_rad_" + str(i)] = tube_rad_vals[i]
        nominal_data["rho_" + str(i)] = rho_vals[i]

    z_high = {'fid_axial': 40, 'fid_radial': 4}
    f = eval_cfd_cylinder


if flag == 'cross':
    data = {}
    for i in range(n_circ):
        for j in range(n_cross_section):
            data["r_" + str(i)+'_'+str(j)] = [0.002, 0.004]
    z_high = {'fid_axial': 40, 'fid_radial': 4}
    f = eval_cfd_cross



if flag == 'standard':
    data = {}
    data["a"] = 0
    data["f"] = 0
    data["re"] = 50
    data["pitch"] = pitch
    data["coil_rad"] = rad
    data["inversion_loc"] = 0

    z_high = {}
    z_high["fid_axial"] = 50
    z_high["fid_radial"] = 4

    f = eval_cfd_basic
    
standard_input = data | z_high

cpus = 1
cpu_vals = derive_cpu_split(cpus)

shutil.copy("mesh_generation/mesh/system/default_decomposeParDict","mesh_generation/mesh/system/decomposeParDict")
replaceAll("mesh_generation/mesh/system/decomposeParDict","numberOfSubdomains 48;","numberOfSubdomains "+str(cpus)+";")
replaceAll("mesh_generation/mesh/system/decomposeParDict","    n               (4 4 3);","    n               ("+str(cpu_vals[0])+" "+str(cpu_vals[1])+" "+str(cpu_vals[2])+");")



f(standard_input)
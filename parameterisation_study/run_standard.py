import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo
from mesh_generation.coil_cylindrical import create_mesh


coils = 3  # number of coils
h = coils * 0.0103  # max height
N = 2 * np.pi * coils  # angular turns (radians)
n = 12  # points to use

x_data = {}
nominal_data = {}

z_vals = np.linspace(0, h, n)
theta_vals = np.linspace(0+np.pi/2, N+np.pi/2, n)
rho_vals = [0.0125 for i in range(n)]
tube_rad_vals = [0.0025 for i in range(n)]
for i in range(n):

    nominal_data["z_" + str(i)] = z_vals[i]
    x_data["z_" + str(i)] = 0
    nominal_data["theta_" + str(i)] = theta_vals[i]
    x_data["theta_" + str(i)] = 0
    nominal_data["tube_rad_" + str(i)] = tube_rad_vals[i]
    x_data["tube_rad_" + str(i)] = 0
    nominal_data["rho_" + str(i)] = rho_vals[i]
    x_data["rho_" + str(i)] = 0

z_high = {'fid_axial': 20.45, 'fid_radial': 7}

standard_input = x_data | z_high

cpus = int(sys.argv[1])
cpu_vals = derive_cpu_split(cpus)

shutil.copy("mesh_generation/mesh/system/default_decomposeParDict","mesh_generation/mesh/system/decomposeParDict")
replaceAll("mesh_generation/mesh/system/decomposeParDict","numberOfSubdomains 48;","numberOfSubdomains "+str(cpus)+";")
replaceAll("mesh_generation/mesh/system/decomposeParDict","    n               (4 4 3);","    n               ("+str(cpu_vals[0])+" "+str(cpu_vals[1])+" "+str(cpu_vals[2])+");")


def eval_cfd(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = "parameterisation_study/standard_simulation"
    create_mesh(
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


eval_cfd(standard_input)
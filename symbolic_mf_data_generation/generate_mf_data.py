from jax import grad, jit, value_and_grad
import sys 
import os 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import jax.numpy as jnp
from utils import *
from utils_plotting import *
from utils_gp import *
from mesh_generation.coil_basic import create_mesh


def gen_data(f, data_path, x_bounds, fid, n_w):

        try:
             data = read_json(data_path)
        except FileNotFoundError:
                # defining joint space bounds
                x_bounds_og = x_bounds.copy()
                samples = sample_bounds(x_bounds, n_w) 
                data = {"data": []}
                for sample in samples:
                        # create sample dict for evaluation
                        sample_dict = sample_to_dict(sample, x_bounds)

                        # preliminary run info 
                        run_info = {
                        "id": "running",
                        "x": sample_dict,
                        "z": fid,
                        "cost": "running",
                        "obj": "running",
                        }
                        data["data"].append(run_info)
                save_json(data, data_path)

        for i in range(len(data["data"])):
                if data['data'][i]['id'] == 'running':
                        sample_dict = data['data'][i]['x']
                        fid = data['data'][i]['z']
                        # perform function evaluation
                        res = f(sample_dict,fid)
                        run_info = {
                        "id": res["id"],
                        "x": sample_dict,
                        "z": fid,
                        "cost": res["cost"],
                        "obj": res["obj"],
                        }
                        data["data"][i] = run_info
                        # save to file
                        save_json(data, data_path)
        return 


x_bounds = {}
x_bounds["a"] = [0.001, 0.008]
x_bounds["f"] = [2, 8]
x_bounds["re"] = [10, 50]
x_bounds["pitch"] = [0.0075, 0.02]
x_bounds["coil_rad"] = [0.0035, 0.0125]

try:
    data_path = str(sys.argv[1])
    cpus = int(sys.argv[2])
    fid_rad = int(sys.argv[3])
    fid_ax = int(sys.argv[4])
    n_s = int(sys.argv[5])
except IndexError:
    data_path = 'symbolic_mf_data_generation/low/data.json'
    cpus = 2
    fid_rad = 3
    fid_ax = 20
    n_s = 80

cpu_vals = derive_cpu_split(cpus)

shutil.copy("mesh_generation/mesh/system/default_decomposeParDict","mesh_generation/mesh/system/decomposeParDict")
replaceAll("mesh_generation/mesh/system/decomposeParDict","numberOfSubdomains 48;","numberOfSubdomains "+str(cpus)+";")
replaceAll("mesh_generation/mesh/system/decomposeParDict","    n               (4 4 3);","    n               ("+str(cpu_vals[0])+" "+str(cpu_vals[1])+" "+str(cpu_vals[2])+");")


try:
    print('Building simulation folder')
    os.mkdir(data_path.split("data.json")[0] + "simulations/")
except FileExistsError:
    print('Simulation folder already exists')


def eval_cfd(x,fid):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    case = data_path.split('data.json')[0]+'simulations/'+ID
    x = x | fid
    x['inversion_loc'] = 0 
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


# z_bounds["fid_axial"] = [19.51, 50.49]
# z_bounds["fid_radial"] = [1.51, 5.49]

gen_data(eval_cfd, data_path, x_bounds, {"fid_axial": fid_ax,"fid_radial":fid_rad}, n_s)
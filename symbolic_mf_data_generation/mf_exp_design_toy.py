import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main_ed import mfed
from mesh_generation.coil_basic import create_mesh
import uuid

x_bounds = {}
x_bounds["x1"] = [0, 4]

z_bounds = {}
z_bounds["z1"] = [0, 1]


def eval(x: dict):
    x1 = x["x1"]
    z1 = x["z1"]

    f1 = np.sin(x1) - np.sqrt(x1) + 1  # low fidelity
    f2 = np.cos(x1) * (x1**2 - x1) / (x1 + 0.5) + x1 + 1  # high fidelity

    f = (
        (f1 * (1 - z1 * x1)) + (np.random.uniform(-(1 - z1) / 3, (1 - z1) / 3))
    ) + f2 * z1

    return {"obj": f, "cost": z1 + x1 * 0.25, "id": str(uuid.uuid4())}


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
                        sample_dict['z1'] = fid
                        # perform function evaluation
                        res = f(sample_dict)
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

gen_data(eval, "symbolic_mf_data_generation/toy/hf_data.json", x_bounds, 1, 100)

# mfed(
#     eval,
#     "symbolic_mf_data_generation/toy/data.json",
#     x_bounds,
#     z_bounds,
#     120,
#     gamma=0.25,
#     sample_initial=4,
#     int_fidelities=[False],
# )



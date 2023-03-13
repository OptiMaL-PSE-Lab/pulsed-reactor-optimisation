import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from distutils.dir_util import copy_tree

x_bounds = {}
x_bounds["a"] = [0.001, 0.008]
x_bounds["f"] = [2, 8]
x_bounds["re"] = [10, 50]
data_path = "monitoring/data.json"
try:
    print('Building simulation folder')
    os.mkdir(data_path.split("data.json")[0] + "simulations/")
except FileExistsError:
    print('Simulation folder already exists')

s = sample_bounds(x_bounds,32)
data = {}
for s_i in s:
    sample = sample_to_dict(s_i,x_bounds)
    ID = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    case = data_path.split("data.json")[0] + "simulations/" + ID
    copy_tree("mesh_generation/steady_state_case", case)
    parse_conditions_given(case, sample['a'], sample['f'], sample['re'])
    times, values = run_cfd(case)
    data[ID] = sample
    save_json(data,data_path)

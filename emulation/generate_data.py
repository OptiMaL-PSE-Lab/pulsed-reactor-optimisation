import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from distutils.dir_util import copy_tree

x_bounds = {}
x_bounds["nu"] = [5e-7, 2e-6]
x_bounds["vel"] = [0.005,0.02]


data_path = "emulation/data.json"
# try:
#     print('Building simulation folder')
#     os.mkdir(data_path.split("data.json")[0] + "simulations/")
# except FileExistsError:
#     print('Simulation folder already exists')

n = 1
s = sample_bounds(x_bounds,n)
save_json({'samples':list([list(si) for si in s])},'emulation/samples.json')
data = {}
for s_i in s:
    # define sample
    sample = sample_to_dict(s_i,x_bounds)
    # create file ID
    ID = str(uuid4())
    case = data_path.split("data.json")[0] + "simulations/" + ID
    # duplicate case
    copy_tree("mesh_generation/steady_state_case", case)
    # redefine inlet velocity
    velBC = ParsedParameterFile(path.join(case, "0", "U"))
    velBC["boundaryField"]["inlet"]["value"].setUniform(Vector(sample['vel'], 0, 0))
    velBC.writeFile()
    # redefine fluid viscosity
    viscos = ParsedParameterFile(path.join(case,"constant","transportProperties"))
    viscos["nu"][1] = sample['nu']
    viscos.writeFile()
    # run case using simpleFOAM
    run_cfd_simple(case)
    list = os.listdir(case)
    for l in list:
        try:
            t = int(l)
        except:
            continue
        if t != 0:
            break 
    copy_tree(case+"/"+str(t),'emulation/simulations/'+ID+'_r') 
    data[str(ID)] = sample
    shutil.rmtree(case)

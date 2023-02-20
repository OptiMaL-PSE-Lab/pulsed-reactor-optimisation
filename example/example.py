import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from main import mfbo


def f(input):
    # decision variables
    x1 = input['x1']
    x2 = input['x2']
    x3 = input['x3']
    x = [x1,x2,x3]

    # fidelities
    z1 = input['z1']
    z2 = input['z2']


    output = (z1 * x[0])**2 + (z2+x[1])**2 + x[2]**2

    cost = (z1/50 * z2*2) + x2 + 10 # fake computational cost 
    
    return {"obj": -output + 20, "cost": cost, "id": str(uuid4())}


x_bounds = {}
x_bounds["x1"] = [-2,2]
x_bounds["x2"] = [-2,2]
x_bounds["x3"] = [-2,2]

z_bounds = {}
z_bounds["z1"] = [0.8,1.2]
z_bounds["z2"] = [0.8,1.2]

data_path = "example/data.json"

mfbo(
    f,
    data_path,
    x_bounds,
    z_bounds,
    gp_ms = 1,
    opt_ms = 4,
    time_budget=100*60,
    sample_initial=16
)


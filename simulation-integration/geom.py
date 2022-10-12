from bayes_opt_with_constraints.bayes_opt import BayesianOptimization, UtilityFunction
from datetime import datetime
from utils import newJSONLogger, vel_calc, parse_conditions, run_cfd, calculate_N
from bayes_opt_with_constraints.bayes_opt.event import Events
import json
import os 
import numpy.random as rnd
import numpy as np 
from uuid import uuid4
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mesh_generation.coil_basic import create_mesh
import shutil 

def LHS(bounds,p):
    d = len(bounds)
    sample = np.zeros((p,len(bounds)))
    for i in range(0,d):
        sample[:,i] = np.linspace(bounds[i,0],bounds[i,1],p)
        rnd.shuffle(sample[:,i])
    return sample 


def eval_cfd(a, f, re, coil_rad, pitch, inversion_loc):
    fid = [1,1]
    tube_rad = 0.0025
    length = 0.0753
    identifier = identifier = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('Starting to mesh '+identifier)
    newcase = "outputs/geom/" + identifier
    create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, fid,path=newcase,validation=False,build=True)
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    for i in range(16):
        shutil.rmtree(newcase+'/processor'+str(i))
    return N

logger = newJSONLogger(path="outputs/geom/logs.json")

# defining utility function
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

# setting up the optimisation problem
optimizer = BayesianOptimization(
    f=eval_cfd,
    pbounds={
        "a": (0.001, 0.008),
        "f": (2, 8),
        "re": (10, 50),
        "coil_rad": (0.005, 0.0125),
        "pitch": (0.0075, 0.015),
        "inversion_loc": (0,1)
    },
    pcons=[],
    verbose=2
)

# Opening JSON file
# assign logger to optimizer

try:
    logs = []
    with open("outputs/geom/logs.json") as f:
        for line in f:
            logs.append(json.loads(line))

    for log in logs:
        optimizer.register(params=log["params"], target=log["target"])

    iteration = len(logs)



except FileNotFoundError:
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    lb = np.array([0.001,2,10,0.005,0.0075,0])
    ub = np.array([0.008,8,50,0.0125,0.015,1])
    bounds = np.array([lb,ub]).T
    n_init = 12
    init_points = LHS(bounds,n_init)
    keys = ['a','f','re','coil_rad','pitch','inversion_loc']
    for p in init_points:
        p_dict = {}
        for i in range(len(keys)):
            p_dict[keys[i]] = p[i]
        target = eval_cfd(**p_dict)
        optimizer.register(params=p_dict, target=target)
    pass

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
while True:
    next_point = optimizer.suggest(utility)
    target = eval_cfd(**next_point)
    optimizer.register(params=next_point, target=target)

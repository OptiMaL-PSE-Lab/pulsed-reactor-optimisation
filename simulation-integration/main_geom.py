from bayes_opt_with_constraints.bayes_opt import BayesianOptimization, UtilityFunction
from utils import newJSONLogger
from bayes_opt_with_constraints.bayes_opt.event import Events
import json
from datetime import datetime
import os 
import numpy as np 

logger = newJSONLogger(path="outputs/geom/logs.json")

# defining utility function
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

def eval_cfd(a, f, re, coil_rad, pitch):
    tube_rad = 0.0025
    length = 0.0785
    fid = [1,1]
    inversion_loc = None
    identifier = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    print('Starting to mesh '+identifier)
    newcase = "outputs/geom/" + identifier
    create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, fid,path=newcase,validation=False,build=True)
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    return N

# setting up the optimisation problem
optimizer = BayesianOptimization(
    f=eval_cfd,
    pbounds={
        "a": (0.001, 0.008),
        "f": (2, 8),
        "re": (10, 50),
        "coil_rad": (0.003, 0.0125),
        "pitch": (0.0075, 0.015),
    },
    pcons=[],
    verbose=2,
    random_state=1,
)


# Opening JSON file
try:
    logs = []
    with open("outputs/geom/logs.json") as f:
        for line in f:
            logs.append(json.loads(line))

    for log in logs:
        optimizer.register(params=log["params"], target=log["target"])
except FileNotFoundError:
    pass
# assign logger to optimizer
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
lb = np.array([0.001,2,10,0.003,0.0075])
ub = np.array([0.008,8,50,0.0125,0.015])
n_init = 15
init_points = np.random.uniform(0,1,(len(lb),n_init))
init_points = np.array([[(init_points[i,j] + lb[i]) *(ub[i]-lb[i]) for i in range(len(lb))]for j in range(n_init)])

keys = ['a','f','re','coil_rad','pitch']
for p in init_points:
    p_dict = {}
    for i in range(len(keys)):
        p_dict[keys[i]] = p[i]
    target = eval_cfd(**p_dict)
    optimizer.register(params=p_dict, target=target)

while True:
    next_point = optimizer.suggest(utility)
    print(next_point)
    target = eval_cfd(**next_point)
    optimizer.register(params=next_point, target=target)

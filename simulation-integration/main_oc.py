from bayes_opt_with_constraints.bayes_opt import BayesianOptimization, UtilityFunction
from utils import eval_cfd_operating_conditions, newJSONLogger
from bayes_opt_with_constraints.bayes_opt.event import Events
import json


logger = newJSONLogger(path="/output/logs_oc.json")

# defining utility function
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)


def pcon(x):
    return 1000000 - (x[0] * 1000 + x[1])


pcon_dict = {"type": "ineq", "fun": pcon}

# setting up the optimisation problem
optimizer = BayesianOptimization(
    f=eval_cfd_operating_conditions,
    pbounds={"a": (0.001, 0.008), "f": (2, 8), "re": (10, 50)},
    pcons=pcon_dict,
    verbose=2,
    random_state=1,
)


# Opening JSON file
try:
    logs = []
    with open("simulation-integration/output/logs_oc.json") as f:
        for line in f:
            logs.append(json.loads(line))

    for log in logs:
        optimizer.register(params=log["params"], target=log["target"])
except FileNotFoundError:
    pass
# assign logger to optimizer
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

while True:
    next_point = optimizer.suggest(utility)
    target = eval_cfd_operating_conditions(**next_point)
    optimizer.register(params=next_point, target=target)

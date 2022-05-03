
from bayes_opt import BayesianOptimization, UtilityFunction
from utils import * 
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# starting logger for bayes opt that doesn't reset

class newJSONLogger(JSONLogger) :
      def __init__(self, path):
            self._path=None
            super(JSONLogger, self).__init__()
            self._path = path if path[-5:] == ".json" else path + ".json"
logger = newJSONLogger(path="./logs.json")

# defining utility function 
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

# setting up the optimisation problem
optimizer = BayesianOptimization(
        f=eval_cfd,
        pbounds={
            "a": (0.001, 0.008),
            "f": (2,8),
            "re": (10,50)
        },
        verbose=2,
        random_state=1,
    )

# assign logger to optimizer
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=0,
    n_iter=1000,
)




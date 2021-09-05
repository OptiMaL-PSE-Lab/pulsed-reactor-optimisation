from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np
from utils import read_data, discretise_design, CFD_function

data = read_data()  # reading data from csv
inputs = data[
    ["Design", "Dt (mm)", "Rc (mm)", "P (mm)", "V (mL)", "Ren", "f (Hz)", "xo (mm)"]
]  # separating input columns
outputs = data[["Navg"]]  #  obtaining output column
inputs = inputs.to_dict(orient="records")  # producing a list of dictionaries
outputs = outputs.values[:, 0]  # producing a list of outputs

# defining an aquisition function
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
# setting up the optimisation problem
optimizer = BayesianOptimization(
    f=CFD_function,
    pbounds={
        "Design": (1, 6),
        "Dt (mm)": (5, 7.5),
        "Rc (mm)": (12.5, 32.5),
        "P (mm)": (7.5, 12.5),
        "V (mL)": (16, 18),
        "Ren": (10, 50),
        "f (Hz)": (0, 10),
        "xo (mm)": (0, 10),
    },
    verbose=2,
    random_state=1,
)

# initialising Gaussian processes with existing data
for i in range(len(outputs)):
    optimizer.register(params=inputs[i], target=outputs[i])

while True:
    # obtaining the next suggested point
    next_point = optimizer.suggest(utility)
    # discretising the design variable
    next_point = discretise_design(next_point)
    # evaluating the set of inputs
    output = CFD_function(next_point)
    print("Current objective function: ", np.round(output, 3))
    # adding the point and its output to the GP
    optimizer.register(params=next_point, target=output)

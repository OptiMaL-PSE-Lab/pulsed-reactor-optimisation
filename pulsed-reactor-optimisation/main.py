from bayes_opt import BayesianOptimization, UtilityFunction
import pandas as pd
from utils import read_data, discretise_design, CFD_function


def variable_suggestion():
    data = read_data()  # reading data from csv
    generated_data = pd.read_csv("data/helical_coil_data_generated.csv")
    var_list = [
        "Design",
        "Dt (mm)",
        "Rc (mm)",
        "P (mm)",
        "V (mL)",
        "Ren",
        "f (Hz)",
        "xo (mm)",
    ]
    inputs = data[var_list]  # separating input columns
    generated_inputs = generated_data[var_list]
    outputs = data[["Navg"]]  #  obtaining output column
    generated_outputs = generated_data[["Navg"]]
    inputs = inputs.append(generated_inputs)
    outputs = outputs.append(generated_outputs)
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

    next_point = optimizer.suggest(utility)
    next_point = discretise_design(next_point)
    print(next_point)
    df = pd.DataFrame(next_point, index=[0])
    generated_data = generated_data.append(df)
    generated_data.to_csv("data/helical_coil_data_generated.csv", index=False)
    return


variable_suggestion()

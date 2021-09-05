import numpy as np
import pandas as pd


def read_data():
    # reading csv file
    dataframe = pd.read_csv("data/helical_coil_data_clean.csv")
    designs = dataframe["Design"].values
    # removing 'R' from design values
    designs = [int(i.split("R")[-1]) for i in designs]
    dataframe["Design"] = designs
    return dataframe


def discretise_design(dataframe):
    dataframe["Design"] = int(np.round((dataframe["Design"]), 0))
    return dataframe


def CFD_function(inputs):
    # code to evaluate CFD here
    target = 0
    return target

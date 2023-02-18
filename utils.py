import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from datetime import datetime
from scipy.special import factorial
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import cm
from os import path
import shutil
import sys
from scipy.optimize import minimize
import matplotlib.colors as colors
from uuid import uuid4
import pickle
import time 
import jax.numpy as jnp 
from distutils.dir_util import copy_tree
from scipy.signal import find_peaks
from PyFoam.Basics.DataStructures import Vector
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.AnalyzedRunner import AnalyzedRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.LogAnalysis.SimpleLineAnalyzer import GeneralSimpleLineAnalyzer
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
import numpy.random as rnd

import gpjax as gpx
from jax import grad, jit, value_and_grad
import jax.numpy as jnp
from uuid import uuid4
import jax.random as jr
import optax as ox
import jax
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

HPC = True


def format_data(data):
    inputs = []
    obj = []
    cost = []
    for d in data["data"]:
        if d["id"] != "running":
            inputs += [list(d["x"].values())]
            obj += [d["obj"]]
            cost += [d["cost"]]

    inputs = jnp.array(inputs)
    obj = jnp.array(obj).reshape(-1, 1)
    cost = jnp.array(cost).reshape(-1, 1)
    return inputs, obj, cost


def mean_std(vec):
    return np.mean(vec, axis=0), np.std(vec, axis=0)


def normalise(vec, mean, std):
    return (vec - mean) / std


def unnormalise(vec, mean, std):
    return (vec * std) + mean


def normalise_bounds_dict(bounds, mean, std):
    keys = list(bounds.keys())
    new_bounds = {}
    for i in range(len(keys)):
        original_bounds = np.array(bounds[keys[i]])
        normalised_bounds = (original_bounds - mean[i]) / std[i]
        new_bounds[keys[i]] = list(normalised_bounds)
    return new_bounds



def sample_bounds(bounds, n):
    sample = lhs(jnp.array(list(bounds.values())), n,log=False)
    return sample


def sample_to_dict(sample, bounds):
    sample_dict = {}
    keys = list(bounds.keys())
    for i in range(len(sample)):
        sample_dict[keys[i]] = float(sample[i])
    return sample_dict


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
    return


def lhs(bounds: list, p: int,log):
    d = len(bounds)
    sample = np.zeros((p, len(bounds)))
    for i in range(0, d):
        if log is False:
            sample[:, i] = np.linspace(bounds[i, 0], bounds[i, 1], p)
        else:
            sample[:, i] = np.geomspace(bounds[i, 0], bounds[i, 1], p)
        rnd.shuffle(sample[:, i])
    return sample



def calc_etheta(N: float, theta: float) -> float:
    z = factorial(N - 1)
    xy = (N * ((N * theta) ** (N - 1))) * (np.exp(-N * theta))
    etheta_calc = xy / z
    return etheta_calc 


def loss(N: list, theta: list, etheta: list) -> float:
    et = []
    for i in range(len(etheta)):
        et.append(calc_etheta(N, theta[i]))
    error_sq = (max(etheta) - max(et)) ** 2
    return error_sq


class CompactAnalyzer(BoundingLogAnalyzer):
    def __init__(self):
        BoundingLogAnalyzer.__init__(self)
        self.addAnalyzer(
            "concentration",
            GeneralSimpleLineAnalyzer(
                r"averageConcentration", r"^[ ]*areaAverage\(outlet\) of s = (.+)$"
            ),
        )


def vel_calc(re):
    return (re * 9.9 * 10**-4) / (990 * 0.005)


def val_to_rtd(time, value, path):
    value = np.array(value)
    time = np.array(time)
    peaks, _ = find_peaks(value, prominence=0.00001)
    times_peaks = time[peaks]
    values_peaks = value[peaks]

    # m = np.argmax(value)
    # value = np.append(value[:m],np.flip(value[:m]))
    # time = np.linspace(time[0],(time[m])*2,len(value))
    # peaks, _ = find_peaks(value, prominence=0.0001)
    # times_peaks = time[peaks]
    # values_peaks = value[peaks]
    # plt.plot(times_peaks, values_peaks, c="b", lw=1,label='Un-skew')

    if path != None:
        plt.figure()
        plt.plot(time, value, c="k", lw=1, alpha=0.1)
        plt.plot(times_peaks, values_peaks, c="r", lw=1, label="CFD")
        plt.xlabel("time")
        plt.ylabel("concentration")
        plt.legend()
        plt.savefig(path + "/preprocessed_plot.png")

    # difference between time values
    dt = np.diff(times_peaks)[0]

    # getting lists of interest (theta, e_sheta)
    et = values_peaks / (sum(values_peaks * dt))
    tau = (sum(times_peaks * values_peaks * dt)) / sum(values_peaks * dt)
    etheta = tau * et
    theta = times_peaks / tau
    return theta, etheta


def calculate_N(value, time, path):
    # obtaining a smooth curve by taking peaks
    theta, etheta = val_to_rtd(time, value, path)
    # fitting value of N
    s = 10000
    n0_list = np.logspace(np.log(1), np.log(50), s)

    best = np.Inf
    for n0 in n0_list:
        l = loss(n0, theta, etheta)
        if l < best:
            best = l
            N = n0


    if path == None:
        return N 
    else:
        plt.figure()
        plt.scatter(theta, etheta, c="k", alpha=0.4, label="CFD")
        etheta_calc = []
        for t in theta:
            etheta_calc.append(calc_etheta(N, t))
        plt.plot(theta, etheta_calc, c="k", ls="dashed", label="Dimensionless")
        plt.legend()
        plt.xlim(0, 4)
        plt.ylim(0, 2.5)
        plt.savefig(path + "/dimensionless_conversion.png")
        return N


def parse_conditions_geom_only(case, vel):
    velBC = ParsedParameterFile(path.join(case, "0", "U"))
    # velBC["boundaryField"]["inlet"]["variables"][1] = '"amp= %.5f;"' % a
    # velBC["boundaryField"]["inlet"]["variables"][0] = '"freq= %.5f;"' % f
    velBC["boundaryField"]["inlet"]["variables"][2] = '"vel= %.5f;"' % vel
    velBC["boundaryField"]["inlet"]["value"].setUniform(Vector(vel, 0, 0))
    velBC.writeFile()
    decomposer = UtilityRunner(
        argv=["decomposePar", "-case", case],
        logname="decomposePar",
    )
    decomposer.start()
    return


def parse_conditions(case, x):
    a = x["a"]
    f = x["f"]
    vel = vel_calc(x["re"])

    velBC = ParsedParameterFile(path.join(case, "0", "U"))
    velBC["boundaryField"]["inlet"]["variables"][1] = '"amp= %.5f;"' % a
    velBC["boundaryField"]["inlet"]["variables"][0] = '"freq= %.5f;"' % f
    velBC["boundaryField"]["inlet"]["variables"][2] = '"vel= %.5f;"' % vel
    velBC["boundaryField"]["inlet"]["value"].setUniform(Vector(vel, 0, 0))
    velBC.writeFile()
    decomposer = UtilityRunner(
        argv=["decomposePar", "-case", case],
        logname="decomposePar",
    )
    decomposer.start()
    return


def run_cfd(case):
    if HPC is True:
        run_command = f"mpiexec pimpleFoam -parallel"
    else:
        run_command = f"pimpleFoam"
    run = AnalyzedRunner(
        CompactAnalyzer(),
        argv=[run_command, "-case", case],
        logname="Solution",
    )
    # running CFD
    run.start()
    # post processing concentrations
    times = run.getAnalyzer("concentration").lines.getTimes()
    values = run.getAnalyzer("concentration").lines.getValues("averageConcentration_0")
    time = np.array(times)  # list of times
    value = np.array(values)  # list of concentrations
    return time, value


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def list_from_dict(list_of_dicts,key):
    l = []
    for d in list_of_dicts:
        try:
            l.append(d[key])
        except KeyError:
            l.append([np.nan])
    return l 


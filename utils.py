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
import fileinput
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

parallel = False

def format_data(data):
    # Reads a data file and returns inputs, outputs, costs

    # initialise lists
    inputs = []
    obj = []
    cost = []
    # iterate over data
    for d in data["data"]:
        # check the solution isn't still running
        if d["id"] != "running":
            # append values to lists
            inputs += [list(d["x"].values())]
            obj += [d["obj"]]
            cost += [d["cost"]]

    # reformat lists to correct shaped arrays
    inputs = jnp.array(inputs)
    obj = jnp.array(obj).reshape(-1, 1)
    cost = jnp.array(cost).reshape(-1, 1)

    return inputs, obj, cost


def mean_std(vec):
    # calculates the mean and standard deviation of a vector
    return np.mean(vec, axis=0), np.std(vec, axis=0)


def normalise(vec, mean, std):
    # normalises a vector or matrix using mean and standard deviation
    return (vec - mean) / std


def unnormalise(vec, mean, std):
    # unnormalises a vector or matrix using mean and standard deviation
    return (vec * std) + mean


def normalise_bounds_dict(bounds, mean, std):
    # normalises a bounds dictionary 

    # get keys of bounds dictionary
    keys = list(bounds.keys())
    new_bounds = {}
    for i in range(len(keys)):
        # get list of upper and lower bounds for each variable
        original_bounds = np.array(bounds[keys[i]])
        # normalise these values
        normalised_bounds = (original_bounds - mean[i]) / std[i]
        # replace original dict with new normalised dict values
        new_bounds[keys[i]] = list(normalised_bounds)
    return new_bounds


def sample_bounds(bounds, n):
    # given a bounds dict, sample n solutions using LHS

    # note: default is NON geometric LHS here
    sample = lhs(jnp.array(list(bounds.values())), n, log=False)
    return sample


def sample_to_dict(sample, bounds):
    # convert a list of values to a dictionary 
    # using respective bounds keys

    sample_dict = {}
    keys = list(bounds.keys())
    for i in range(len(sample)):
        sample_dict[keys[i]] = float(sample[i])
    return sample_dict


def read_json(path):
    # read a json file as a dictionary
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    # save a dictionary as a json file
    with open(path, "w") as f:
        json.dump(data, f)
    return


def lhs(bounds: list, p: int, log):
    # latin hypercube sampling 

    d = len(bounds)
    sample = np.zeros((p, len(bounds)))
    for i in range(0, d):
        # log allows for a geometric sample
        # that is equally spaced in log coordinates
        if log is False:
            sample[:, i] = np.linspace(bounds[i, 0], bounds[i, 1], p)
        else:
            sample[:, i] = np.geomspace(bounds[i, 0], bounds[i, 1], p)
        rnd.shuffle(sample[:, i])
    return sample


def calc_etheta(N: float, theta: float) -> float:
    # calculate dimensionless concentration from time and tanks-in-series
    z = factorial(N - 1)
    xy = (N * ((N * theta) ** (N - 1))) * (np.exp(-N * theta))
    etheta_calc = xy / z
    return etheta_calc


def loss(N: list, theta: list, etheta: list) -> float:
    # quantifies the difference between true dimensionless concentration
    # and predicted values from a equivalent tanks-in-series
    et = []
    for i in range(len(etheta)):
        et.append(calc_etheta(N, theta[i]))

    # I found this is most robust by quantifying the loss as: 
    error_sq = (max(etheta) - max(et)) ** 2
    return error_sq

def loss_sq(N: list, theta: list, etheta: list) -> float:
    # quantifies the difference between true dimensionless concentration
    # and predicted values from a equivalent tanks-in-series
    et = []
    for i in range(len(etheta)):
        et.append(calc_etheta(N, theta[i]))

    # I found this is most robust by quantifying the loss as: 
    error_sq = sum((etheta[i] - et[i]) ** 2 for i in range(len(etheta)))/len(etheta)
    return (max(etheta) - max(et)) ** 2,error_sq


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
    # calculate velocity from reynolds number
    return (re * 9.9 * 10**-4) / (990 * 0.005)


def val_to_rtd(time, value, path):
    # convert measured values of concentration and time
    # to dimensionless concentration and time

    value = np.array(value)
    time = np.array(time)
    # periodic output so find only the peaks, tol can be changed
    peaks, _ = find_peaks(value, prominence=0.00001)
    times_peaks = time[peaks]
    values_peaks = value[peaks]

    # plot this if you want 
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
    s = 5000
    n0_list = np.logspace(np.log(1), np.log(50), s)

    # forgo any optimisation here because this is more robust
    best = np.Inf
    for n0 in n0_list:
        l = loss(n0, theta, etheta)
        if l < best:
            best = l
            N = n0

    print(best)
    # plot this is you want 
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

def val_to_rtd_clean(time, value):
    # convert measured values of concentration and time
    # to dimensionless concentration and time

    value = np.array(value)
    time = np.array(time)
    # periodic output so find only the peaks, tol can be changed
    # difference between time values
    dt = np.mean(np.diff(time))
    # getting lists of interest (theta, e_sheta)
    et = value / (sum(value * dt))
    tau = (sum(time * value * dt)) / sum(value * dt)
    etheta = tau * et
    theta = time / tau
    return theta, etheta


def calculate_N_clean(value, time, path):

    t_d, et_d = val_to_rtd_clean(time, value)
    s = 10000
    n0_list = np.logspace(np.log(1), np.log(100), s)

    # forgo any optimisation here because this is more robust
    best = np.Inf
    for n0 in n0_list:
        l,le = loss_sq(n0, t_d, et_d)
        if le < best:
            best = le
            N = n0
    best = best * 100
    # plot this is you want 
    if path == None:
        return N, best
    else:
        plt.figure()
        plt.plot(t_d, et_d, c="k", label="CFD")
        etheta_calc = []
        for t in t_d:
            etheta_calc.append(calc_etheta(N, t))
        plt.plot(t_d, etheta_calc, c="k", ls="dashed", label="Tanks-in-series")
        plt.legend()
        plt.savefig(path + "/dimensionless_conversion.png")
        return N, best

def parse_conditions(case, x):
    # append operating conditions to correct location in case file

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


def parse_conditions_given(case, a, f, re):
    # append given operating conditions (no dictionary needed)
    vel = vel_calc(re)

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
    # run a casefile

    # multiple procs if true
    if parallel is True:
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


def list_from_dict(list_of_dicts, key):
    # returns a list of values from a list of dictionarys and 
    # a given key 
    l = []
    for d in list_of_dicts:
        try:
            l.append(d[key])
        except KeyError:
            l.append([np.nan])
    return l


def is_prime(x):
    if x < 2:
        return False
    for i in range(2, x):
        if x % i == 0:
            return False
    return True

def is_divisible_by(x,n):
    return x % n == 0

def derive_cpu_split(x):
    splits = [x,1,1]

    for v in np.flip([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]):
        while True: 
            if is_divisible_by(splits[0],v):
                splits[0] = int(splits[0] / v)
                splits[1] *= v
            else:
                break 

        while True: 
            if is_divisible_by(splits[1],v) and splits[1] > splits[2]:
                splits[1] = int(splits[1] / v)
                splits[2] *= v
            else:
                break 

    for v in np.flip([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]):
        if is_divisible_by(splits[-1],v) and splits[0] == 1:
            splits[-1] = int(splits[-1] / v)
            splits[0] *= v

    for i in range(len(splits)):
        splits[i] = int(splits[i])
    return splits


def replaceAll(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp,replaceExp)
        sys.stdout.write(line)
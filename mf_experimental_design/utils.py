import numpy as np
from datetime import datetime
from scipy.special import factorial
import matplotlib.pyplot as plt
import os
from os import path
import shutil
import sys
from uuid import uuid4
import pickle
from distutils.dir_util import copy_tree
from scipy.signal import find_peaks
from PyFoam.Basics.DataStructures import Vector
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.AnalyzedRunner import AnalyzedRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.LogAnalysis.SimpleLineAnalyzer import GeneralSimpleLineAnalyzer
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
import numpy.random as rnd 

try:
    HPC = str(sys.argv[1])
    if HPC == "HPC":
        HPC = True

    else:
        HPC = False
except:
    HPC = False



def lhs(bounds: list,p: int):
    d = len(bounds)
    sample = np.zeros((p,len(bounds)))
    for i in range(0,d):
        sample[:,i] = np.linspace(bounds[i,0],bounds[i,1],p)
        rnd.shuffle(sample[:,i])
    return sample 


def calc_etheta(N: float, theta: float, off: float, up: float) -> float:
    theta = theta - off
    z = factorial(N - 1)
    xy = (N * ((N * theta) ** (N - 1))) * (np.exp(-N * theta))
    etheta_calc = xy / z
    return etheta_calc * up


def loss(X: list, theta: list, etheta: list) -> float:
    N, off, up = X
    error_sq = 0
    et = []
    for i in range(len(etheta)):
        et.append(calc_etheta(N, theta[i], off, up))

    # for i in range(len(theta)):
    #     if theta[i] > 2:
    #         error_sq += 0
    #     else:
    #         error_sq += (calc_etheta(N, theta[i], off, up) - etheta[i]) ** 2
    error_sq += (max(etheta)-max(et))**2
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

def val_to_rtd(time,value,path):
    value = np.array(value)
    time = np.array(time)
    plt.figure()
    peaks, _ = find_peaks(value, prominence=0.00001)
    times_peaks = time[peaks]
    values_peaks = value[peaks]
    plt.plot(time, value, c="k", lw=1, alpha=0.1)
    plt.plot(times_peaks, values_peaks, c="r", lw=1,label='CFD')

    # m = np.argmax(value)
    # value = np.append(value[:m],np.flip(value[:m]))
    # time = np.linspace(time[0],(time[m])*2,len(value))
    # peaks, _ = find_peaks(value, prominence=0.0001)
    # times_peaks = time[peaks]
    # values_peaks = value[peaks]
    # plt.plot(times_peaks, values_peaks, c="b", lw=1,label='Un-skew')

    plt.grid()
    plt.xlabel("time")
    plt.ylabel("concentration")
    plt.legend()
    plt.savefig(path+"/preprocessed_plot.png")

    # difference between time values
    dt = np.diff(times_peaks)[0]

    # getting lists of interest (theta, e_sheta)
    et = values_peaks / (sum(values_peaks * dt))
    tau = (sum(times_peaks * values_peaks * dt)) / sum(values_peaks * dt)
    etheta = tau * et
    theta = times_peaks / tau
    return theta,etheta

def calculate_N(value, time,path):
    # obtaining a smooth curve by taking peaks
    theta,etheta = val_to_rtd(time,value,path)
    # fitting value of N
    s = 10000
    x0_list = np.array(
        [
            np.logspace(np.log(1), np.log(50), s),
            np.random.uniform(-0.001, 0.001, s),
            np.random.uniform(1, 1.0001, s),
        ]
    ).T

    best = np.Inf
    for x0 in x0_list:
        l = loss(x0, theta, etheta)
        if l < best:
            best = l
            X = x0

    N, off, up = X

    plt.figure()
    plt.scatter(theta, etheta, c="k", alpha=0.4,label="CFD")
    etheta_calc = []
    for t in theta:
        etheta_calc.append(calc_etheta(N, t, off, up))
    plt.plot(theta, etheta_calc, c="k",ls='dashed', label="Dimensionless")
    plt.grid()
    plt.legend()
    plt.xlim(0,4)
    plt.ylim(0,2.5)
    plt.savefig(path+"/dimensionless_conversion.png")
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
    a = x['a']
    f = x['f']
    vel = vel_calc(x['re'])

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
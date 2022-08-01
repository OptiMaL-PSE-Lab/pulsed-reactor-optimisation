import numpy as np
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
from bayes_opt_with_constraints.bayes_opt.logger import JSONLogger
from PyFoam.Basics.DataStructures import Vector
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.AnalyzedRunner import AnalyzedRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.LogAnalysis.SimpleLineAnalyzer import GeneralSimpleLineAnalyzer
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mesh_generation.coil_basic import create_mesh
from mesh_generation.coil_validation import create_validation_mesh

try:
    HPC = str(sys.argv[1])
    if HPC == "HPC":
        HPC = True

    else:
        HPC = False
except:
    HPC = False


class newJSONLogger(JSONLogger):
    def __init__(self, path):
        self._path = None
        super(JSONLogger, self).__init__()
        self._path = path if path[-5:] == ".json" else path + ".json"


def calc_etheta(N, theta, off, up):
    theta = theta - off
    z = factorial(N - 1)
    xy = (N * ((N * theta) ** (N - 1))) * (np.exp(-N * theta))
    etheta_calc = xy / z
    return etheta_calc * up


def loss(X, theta, etheta):
    N, off, up = X
    error_sq = 0
    for i in range(len(theta)):
        if theta[i] > 2:
            error_sq += 0
        else:
            error_sq += (calc_etheta(N, theta[i], off, up) - etheta[i]) ** 2
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


def calculate_N(value, time,path):
    # obtaining a smooth curve by taking peaks
    peaks, _ = find_peaks(value, prominence=0.01)
    times_peaks = time[peaks]
    values_peaks = value[peaks]

    plt.figure()
    plt.plot(time, value, c="k", lw=1, alpha=0.1)
    plt.plot(times_peaks, values_peaks, c="r", lw=1)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("concentration")
    plt.savefig(path+"/preprocessed_plot.png")

    # difference between time values
    dt = np.diff(times_peaks)[0]

    # getting lists of interest (theta, e_theta)
    et = values_peaks / (sum(values_peaks * dt))
    tau = (sum(times_peaks * values_peaks * dt)) / sum(values_peaks * dt)
    etheta = tau * et
    theta = times_peaks / tau

    # fitting value of N
    s = 100000
    x0_list = np.array(
        [
            np.logspace(np.log(1), np.log(50), s),
            np.random.uniform(-1, 0, s),
            np.random.uniform(0, 1, s),
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
    plt.plot(theta, etheta, c="k", linestyle="dashed", label="CFD")
    etheta_calc = []
    for t in theta:
        etheta_calc.append(calc_etheta(N, t, off, up))
    plt.plot(theta, etheta_calc, c="k", label="Dimensionless")
    plt.grid()
    plt.legend()
    plt.savefig(path+"/dimensionless_conversion.png")
    return N




def parse_conditions(case, a, f, vel):
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
        print("TRUE")
        run_command = f"mpiexec pimpleFoam -parallel"
    else:
        print("FALSE")
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


def eval_cfd(a, f, re, coil_rad, pitch):
    tube_rad = 0.0025
    length = 0.0785
    inversion_loc = None
    identifier = str(uuid4())
    print('Starting to mesh '+identifier)
    newcase = "simulation-integration/output_geom/" + identifier
    create_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, path=newcase)
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    return N

def eval_cfd_validation(a, f, re, coil_rad, pitch,tube_rad,length):
    inversion_loc = None
    identifier = str(uuid4())
    print('Starting to mesh '+identifier)
    newcase = "simulation-integration/output_validation/" + identifier
    create_validation_mesh(coil_rad, tube_rad, pitch, length, inversion_loc, path=newcase)
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    return N,time,value


def eval_cfd_operating_conditions(a, f, re):
    identifier = str(uuid4())
    print('Starting to mesh '+identifier)
    newcase = "simulation-integration/output_oc/" + identifier
    vel = vel_calc(re)
    parse_conditions(newcase, a, f, vel)
    time, value = run_cfd(newcase)
    N = calculate_N(value, time,newcase)
    return N

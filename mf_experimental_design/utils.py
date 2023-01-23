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
import matplotlib.colors as colors
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

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)


try:
    HPC = str(sys.argv[1])
    if HPC == "HPC":
        HPC = True

    else:
        HPC = False
except:
    HPC = False


def lhs(bounds: list, p: int):
    d = len(bounds)
    sample = np.zeros((p, len(bounds)))
    for i in range(0, d):
        sample[:, i] = np.linspace(bounds[i, 0], bounds[i, 1], p)
        rnd.shuffle(sample[:, i])
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
    error_sq += (max(etheta) - max(et)) ** 2
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
    plt.figure()
    peaks, _ = find_peaks(value, prominence=0.00001)
    times_peaks = time[peaks]
    values_peaks = value[peaks]
    plt.plot(time, value, c="k", lw=1, alpha=0.1)
    plt.plot(times_peaks, values_peaks, c="r", lw=1, label="CFD")

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
    plt.scatter(theta, etheta, c="k", alpha=0.4, label="CFD")
    etheta_calc = []
    for t in theta:
        etheta_calc.append(calc_etheta(N, t, off, up))
    plt.plot(theta, etheta_calc, c="k", ls="dashed", label="Dimensionless")
    plt.grid()
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


def plot_results(data, crit):
    data = data["data"]
    crit = crit["data"]
    obj = []
    cost = []
    fid = []
    n = 25
    cutoff = True

    if cutoff is True:
        data = data[n:]

    for d in data:
        if d["id"] != "running":
            obj.append(d["obj"])
            cost.append(d["cost"])
            x = d["x"]
            fid_vals = []
            for x_k in list(x.keys()):
                if x_k.split("_")[0] == "fid":
                    fid_vals.append(x[x_k])
            fid.append(fid_vals)
    it = np.arange(len(obj))
    b = 0
    best_obj = []
    for o in obj:
        if o > b:
            best_obj.append(o)
            b = o
        else:
            best_obj.append(b)
    b = 0
    best_crit = []
    for c in crit:
        if c > b:
            best_crit.append(c)
            b = c
        else:
            best_crit.append(b)

    lw = 2
    ms = 60
    grid_alpha = 0.3
    font_size = 15

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    for a in ax.ravel():
    
        a.spines["right"].set_visible(False)
        a.spines["top"].set_visible(False)
        a.tick_params(axis='both', which='major', labelsize=font_size-2)


    im1 = ax[0,0].scatter(it, obj, c=cost, marker="+", s=ms, lw=lw,norm=colors.LogNorm(vmin=min(cost), vmax=max(cost)))
    ax[0,0].plot(it, best_obj, c="k", ls="--", lw=lw,alpha=0.75,label='Best')
    ax[0,0].legend(frameon=False,fontsize=14)
    ax[0,0].grid(alpha=grid_alpha)
    ax[0,0].set_xlabel(r"Iteration", fontsize=font_size)
    ax[0,0].set_ylabel("Tanks-in-series",
        fontsize=font_size,
    )
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cb = fig.colorbar(im1, cax=cax, orientation='vertical')
    cax.tick_params(labelsize=14) 
    cb.set_label(label='Simulation Cost (s)',size=14)


    ax[0,1].scatter(np.arange(len(crit)), crit, c="k", marker="+", s=ms, lw=lw)
    ax[0,1].plot(np.arange(len(crit)), best_crit, c="k", ls="--", lw=lw,alpha=0.75,label='Best')
    ax[0,1].legend(frameon=False,fontsize=14)
    ax[0,1].grid(alpha=grid_alpha)
    ax[0,1].set_xlabel(r"Iteration", fontsize=font_size)
    ax[0,1].set_ylabel(r"$\max_x \quad \mu_t(x,z^\bullet)$", fontsize=font_size)

    time = np.cumsum(cost)
    im2 = ax[1,0].scatter(time, obj, c=cost, marker="+", s=ms, lw=lw,norm=colors.LogNorm(vmin=min(cost), vmax=max(cost)))
    ax[1,0].plot(time, best_obj, c="k", ls="--", lw=lw,alpha=0.75,label='Best')
    ax[1,0].legend(frameon=False,fontsize=14)
    ax[1,0].grid(alpha=grid_alpha)
    ax[1,0].ticklabel_format(style='sci', axis='x',scilimits=(0,4))

    ax[1,0].set_xlabel(r"Wall-clock time (s)", fontsize=font_size)
    ax[1,0].set_ylabel("Tanks-in-series",
        fontsize=font_size,
    )
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cb = fig.colorbar(im2, cax=cax, orientation='vertical')
    cax.tick_params(labelsize=14) 
    cb.set_label(label='Simulation Cost (s)',size=14)

    ax[1,1].scatter(time, crit, c="k", marker="+", s=ms, lw=lw)
    ax[1,1].plot(time, best_crit, c="k", ls="--", lw=lw,alpha=0.75,label='Best')
    ax[1,1].legend(frameon=False,fontsize=14)
    ax[1,1].grid(alpha=grid_alpha)
    ax[1,1].set_xlabel(r"Wall-clock time (s)", fontsize=font_size)
    ax[1,1].ticklabel_format(style='sci', axis='x',scilimits=(0,4))
    ax[1,1].set_ylabel(r"$\max_x \quad \mu_t(x,z^\bullet)$", fontsize=font_size)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    plt.savefig("outputs/mf/res.png", dpi=800)

    return


def plot_fidelities(data):
    z_vals = []
    for d in data["data"]:
        xv = d["x"]
        zv = []
        for xk in list(xv.keys()):
            if xk.split("_")[0] == "fid":
                zv.append(xv[xk])
        z_vals.append(zv)
    z_vals = np.array(z_vals)

    z_vals = z_vals[25:, :]
    color = cm.viridis(np.linspace(0, 1, len(z_vals - 1)))
    fig, axs = plt.subplots(1, 1, figsize=(9, 4))

    sc = axs.scatter(
        z_vals[:, 0], z_vals[:, 1], c=np.arange(len(z_vals)), marker="+", lw=3, s=80
    )
    fig.colorbar(sc, ax=axs, label="$t$")
    axs.set_ylabel("Radial Fidelity", fontsize=12)
    axs.set_xlabel("Axial Fidelity", fontsize=12)
    axs.set_yticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    axs.grid(alpha=0.3)
    fig.subplots_adjust(right=1.05, left=0.075, top=0.95, bottom=0.15)
    # for i, c in zip(range(len(z_vals)-1), color):
    #     axs.plot([z_vals[i,0],z_vals[i+1,0]],[z_vals[i,1],z_vals[i+1,1]],lw=6,color=c,alpha=0.95)
    fig.savefig("outputs/mf/fidelities.png", dpi=800)
    return



# plot_fidelities(data)

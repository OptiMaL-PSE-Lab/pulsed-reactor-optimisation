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
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from mf_experimental_design.utils import *
from mesh_generation.coil_basic import create_mesh

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


def aquisition_function(x, gp, cost_gp, fid_high, gamma, beta):
    cost, cost_var = inference(cost_gp, jnp.array([x]))
    # fixing fidelity
    for i in range(len(fid_high)):
        i += 1
        x = x.at[-i].set(fid_high[-i])
    mean, cov = inference(gp, jnp.array([x]))
    return -((mean[0] + beta * cov[0]) / (gamma * cost[0]))[0]

def greedy_function(x, gp, fid_high):
    # fixing fidelity
    for i in range(len(fid_high)):
        x = jnp.append(x,fid_high[i])
    mean, cov = inference(gp, jnp.array([x]))
    return - mean[0]

def sample_bounds(bounds, n):
    sample = lhs(jnp.array(list(bounds.values())), n)
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


def lhs(bounds: list, p: int):
    d = len(bounds)
    sample = np.zeros((p, len(bounds)))
    for i in range(0, d):
        sample[:, i] = np.linspace(bounds[i, 0], bounds[i, 1], p)
        rnd.shuffle(sample[:, i])
    return sample

def eval_cfd(x: dict):
    start = time.time()
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print("Starting to mesh " + ID)
    case = "outputs/mf/" + ID
    create_mesh(
        x,
        length=0.0753,
        tube_rad=0.0025,
        path=case,
    )
    parse_conditions(case, x)
    times, values = run_cfd(case)
    N = calculate_N(values, times, case)
    for i in range(16):
        shutil.rmtree(case + "/processor" + str(i))
    # shutil.rmtree(newcase)
    end = time.time()
    return {"obj": N, "cost": end - start, "id": ID}


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


def plot_results(data,n):
    data = data["data"]
    
    obj = []
    cost = []
    fid = []
    crit = []
    n = 25
    cutoff = True

    if cutoff is True:
        data = data[n:]

    for d in data:
        if d["id"] != "running":
            obj.append(d["obj"])
            cost.append(d["cost"])
            crit.append(unnormalise(d["pred_g_obj_mean"],d["obj_mean"],d["obj_std"]))
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
    m_alpha=0.8
    mar = "o"
    grid_alpha = 0.0
    font_size = 15

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    for a in ax.ravel():
        # a.spines["right"].set_visible(False)
        # a.spines["top"].set_visible(False)
        a.tick_params(axis='both', which='major', labelsize=font_size-2)


    im1 = ax[0,0].scatter(it, obj, c=cost, marker=mar, s=ms, lw=0,norm=colors.LogNorm(vmin=min(cost), vmax=max(cost)),alpha=m_alpha)
    ax[0,0].plot(it, best_obj, c="k", ls=":", lw=lw,alpha=0.75,label='Best')
    #ax[0,0].legend(frameon=False,fontsize=14)
    ax[0,0].set_xlabel(r"Iteration", fontsize=font_size)
    ax[0,0].set_ylabel("Tanks-in-series",
        fontsize=font_size,
    )
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cb = fig.colorbar(im1, cax=cax, orientation='vertical')
    cax.tick_params(labelsize=14) 
    cb.set_label(label='Simulation Cost (s)',size=14)


    ax[0,1].scatter(np.arange(len(crit)), crit, c="k", marker=mar, s=ms, lw=0,alpha=m_alpha)
    ax[0,1].plot(np.arange(len(crit)), best_crit, c="k", ls=":", lw=lw,alpha=0.75,label='Best')
    #ax[0,1].legend(frameon=False,fontsize=14)
    ax[0,1].set_xlabel(r"Iteration", fontsize=font_size)
    ax[0,1].set_ylabel(r"$\max_x \quad \mu_t(x,z^\bullet)$", fontsize=font_size)

    time = np.cumsum(cost)
    im2 = ax[1,0].scatter(time, obj, c=cost, marker=mar, s=ms, lw=0,norm=colors.LogNorm(vmin=min(cost), vmax=max(cost)),alpha=m_alpha)
    ax[1,0].plot(time, best_obj, c="k", ls=":", lw=lw,alpha=0.75,label='Best')
    #ax[1,0].legend(frameon=False,fontsize=14)
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

    ax[1,1].scatter(time, crit, c="k", marker=mar, s=ms, lw=0,alpha=m_alpha)
    ax[1,1].plot(time, best_crit, c="k", ls=":", lw=lw,alpha=0.75,label='Best')
    #ax[1,1].legend(frameon=False,fontsize=14)
    ax[1,1].set_xlabel(r"Wall-clock time (s)", fontsize=font_size)
    ax[1,1].ticklabel_format(style='sci', axis='x',scilimits=(0,4))
    ax[1,1].set_ylabel(r"$\max_x \quad \mu_t(x,z^\bullet)$", fontsize=font_size)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    plt.savefig("outputs/mf/res.png", dpi=800)

    return

def train_gp(inputs, outputs):
    D = gpx.Dataset(X=inputs, y=outputs)
    kern = gpx.RBF(active_dims=[i for i in range(D.in_dim)])
    prior = gpx.Prior(kernel=kern)
    likelihood = gpx.Gaussian(num_datapoints=D.n)
    posterior = prior * likelihood

    mll = jit(posterior.marginal_log_likelihood(D, negative=True))
    opt = ox.adam(learning_rate=1e-3)
    parameter_state = gpx.initialise(posterior)
    inference_state = gpx.fit(mll, parameter_state, opt, num_iters=10000)

    learned_params, _ = inference_state.unpack()
    return posterior, learned_params, D, likelihood


def inference(gp, inputs):
    posterior = gp["posterior"]
    learned_params = gp["learned_params"]
    D = gp["D"]
    likelihood = gp["likelihood"]
    latent_distribution = posterior(learned_params, D)(inputs)
    predictive_distribution = likelihood(learned_params, latent_distribution)
    predictive_mean = predictive_distribution.mean()
    predictive_cov = predictive_distribution.covariance()
    return predictive_mean, predictive_cov

def build_gp_dict(posterior, learned_params, D, likelihood):
    gp_dict = {}
    gp_dict["posterior"] = posterior
    gp_dict["learned_params"] = learned_params
    gp_dict["D"] = D
    gp_dict["likelihood"] = likelihood
    return gp_dict

import jax.numpy as jnp 

def plot_fidelities(data):
    z_vals = []
    c_vals = []
    for d in data["data"]:
        if d['cost'] != 'running':
            xv = d["x"]
            c_vals.append(d['cost'])
            zv = []
            for xk in list(xv.keys()):
                if xk.split("_")[0] == "fid":
                    zv.append(xv[xk])
            z_vals.append(zv)
    z_vals = np.array(z_vals)


    c_vals = c_vals[25:]
    z_vals = z_vals[25:, :]
    color = cm.viridis(c_vals)
    fig, axs = plt.subplots(1, 1, figsize=(5.5, 4))
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='3%', pad=0.05)

    sc = axs.scatter(
        z_vals[:, 0], z_vals[:, 1], c=c_vals, marker="o", lw=0, s=120,alpha=0.8,
        norm=colors.LogNorm(vmin=500, vmax=5000)
    )

    cb = fig.colorbar(sc, cax=cax, orientation='vertical')
    cax.tick_params(labelsize=14) 
    cb.set_label(label='Simulation Cost (s)',size=14)

    axs.set_ylabel("Radial Fidelity", fontsize=12)
    axs.set_xlabel("Axial Fidelity", fontsize=12)
    axs.set_yticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    fig.tight_layout()
    #fig.subplots_adjust(right=1.05, left=0.075, top=0.95, bottom=0.15)
    # for i, c in zip(range(len(z_vals)-1), color):
    #     axs.plot([z_vals[i,0],z_vals[i+1,0]],[z_vals[i,1],z_vals[i+1,1]],lw=6,color=c,alpha=0.95)
    fig.savefig("outputs/mf/fidelities.png", dpi=800)
    return


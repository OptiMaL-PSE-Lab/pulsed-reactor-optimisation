import os
from os import path
from distutils.dir_util import copy_tree
from re import I
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
from PyFoam.Execution.AnalyzedRunner import AnalyzedRunner
from PyFoam.LogAnalysis.SimpleLineAnalyzer import GeneralSimpleLineAnalyzer
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.special import factorial
import matplotlib.pyplot as plt

class CompactAnalyzer(BoundingLogAnalyzer):
    def __init__(self):
        BoundingLogAnalyzer.__init__(self)
        self.addAnalyzer(
            "concentration",
            GeneralSimpleLineAnalyzer(
                r"averageConcentration", r"^[ ]*areaAverage\(auto2\) of s = (.+)$"
            ),
        )


solver = "pimpleFoam"

for a in [0.003]:

    os.mkdir("pimple-amp%.3f" % a)
    os.mkdir("pimple-amp%.3f/0" % a)
    os.mkdir("pimple-amp%.3f/constant" % a)
    os.mkdir("pimple-amp%.3f/system" % a)

    from_directory = "pimple/0"
    to_directory = "pimple-amp%.3f/0" % a
    copy_tree(from_directory, to_directory)

    from_directory = "pimple/constant"
    to_directory = "pimple-amp%.3f/constant" % a
    copy_tree(from_directory, to_directory)

    from_directory = "pimple/system"
    to_directory = "pimple-amp%.3f/system" % a

    copy_tree(from_directory, to_directory)

    Newcase = "pimple-amp%.3f" % a

    velBC = ParsedParameterFile(path.join(Newcase, "0", "U"))
    velBC["boundaryField"]["auto1"]["variables"][1] = '"amp= %.3f;"' % a
    velBC.writeFile()

    run = AnalyzedRunner(
        CompactAnalyzer(),
        argv=[solver, "-case", Newcase],
        logname="Solution",
    )

    run.start()

    times = np.array(run.getAnalyzer("concentration").lines.getTimes())
    values = np.array(
        run.getAnalyzer("concentration").lines.getValues("averageConcentration_0")
    )

    # times and values are numpy arrays, you can do whatever you want here

    time = np.array(times) # list of times
    value = np.array(values) # list of concentrations 

    # obtaining a smooth curve by taking peaks  
    peaks, _ = find_peaks(value, prominence=0.001) 
    times = time[peaks]
    values = value[peaks]

    # difference between time values
    dt = np.diff(times)[0]

    # getting lists of interest (theta, e_theta)
    et = values / (sum(values * dt))
    tau = (sum(times * values  * dt)) / sum(values * dt)
    etheta = tau * et
    theta = times / tau

    def calc_etheta(N,theta):
        z = factorial(N-1)
        xy = (N * ((N * theta) ** (N - 1))) * (np.exp(-N * theta))
        etheta_calc = xy / z
        return etheta_calc

    def loss(N,theta,etheta):
        for i in range(len(theta)):
            etheta_calc = calc_etheta(N,theta[i])
            error_sq = (etheta_calc - etheta[i])**2
        return error_sq

    N = minimize(loss,x0=35,bounds=((0.1,1000),),args=(theta,etheta)).x

    plt.figure()
    plt.plot(theta,etheta,c='k',linestyle='dashed',label='CFD')
    etheta_calc = [] 
    for t in theta:
        etheta_calc.append(calc_etheta(N,t))
    plt.plot(theta,etheta_calc,c='k',label='Dimensionless')
    plt.grid()
    plt.legend()
    plt.savefig(str(a)+'.png')
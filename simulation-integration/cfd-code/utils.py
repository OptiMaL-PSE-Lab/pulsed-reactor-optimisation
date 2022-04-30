import numpy as np 
from scipy.special import factorial
from PyFoam.LogAnalysis.SimpleLineAnalyzer import GeneralSimpleLineAnalyzer
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
import matplotlib.pyplot as plt 
import os
from os import path
from distutils.dir_util import copy_tree
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.AnalyzedRunner import AnalyzedRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from scipy.signal import find_peaks
from scipy.optimize import minimize
from bayes_opt.logger import JSONLogger
import shutil
from datetime import datetime,timezone

np=8

class newJSONLogger(JSONLogger) :
      def __init__(self, path):
            self._path=None
            super(JSONLogger, self).__init__()
            self._path = path if path[-5:] == ".json" else path + ".json"
	    
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


class CompactAnalyzer(BoundingLogAnalyzer):
    def __init__(self):
        BoundingLogAnalyzer.__init__(self)
        self.addAnalyzer(
            "concentration",
            GeneralSimpleLineAnalyzer(
                r"averageConcentration", r"^[ ]*areaAverage\(outlet\) of s = (.+)$"
            ),
        )



def eval_cfd(a):

    # creating solution folder
    os.mkdir("single-coil-amp%.6f" % a)
    os.mkdir("single-coil-amp%.6f/0" % a)
    os.mkdir("single-coil-amp%.6f/constant" % a)
    os.mkdir("single-coil-amp%.6f/system" % a)

    # copying initial conditions
    from_directory = "single-coil/0"
    to_directory = "single-coil-amp%.6f/0" % a
    copy_tree(from_directory, to_directory)

    # copying constants
    from_directory = "single-coil/constant"
    to_directory = "single-coil-amp%.6f/constant" % a
    copy_tree(from_directory, to_directory)

    # copying cfd system to solution
    from_directory = "single-coil/system"
    to_directory = "single-coil-amp%.6f/system" % a

    copy_tree(from_directory, to_directory)

    Newcase = "single-coil-amp%.6f" % a

    velBC = ParsedParameterFile(path.join(Newcase, "0", "U"))
    velBC["boundaryField"]['inlet']["variables"][1] = '"amp= %.6f;"' % a
    velBC.writeFile()

    decomposer= UtilityRunner(
    argv=["decomposePar", "-case",Newcase],
    logname="decomposePar",
    )
    decomposer.start()

    np_substring= "mpiexec" #if HPC
    #np_substring= "mpirun -np {np}" if not HPC
    run_command = f"{np_substring} pimpleFoam -parallel"
    run = AnalyzedRunner(
        CompactAnalyzer(),
        argv=[run_command, "-case", Newcase],
        logname="Solution",
    )
    
    # running CFD 
    run.start()

    
    # post processing concentrations
    times = np.array(run.getAnalyzer("concentration").lines.getTimes())
    values = np.array(
        run.getAnalyzer("concentration").lines.getValues("averageConcentration_0")
    )

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

    # fitting value of N 
    N = minimize(loss,x0=35,bounds=((0.1,1000),),args=(theta,etheta)).x

    plt.figure()
    plt.plot(theta,etheta,c='k',linestyle='dashed',label='CFD')
    etheta_calc = [] 
    for t in theta:
        etheta_calc.append(calc_etheta(N,t))
    plt.plot(theta,etheta_calc,c='k',label='Dimensionless')
    plt.grid()
    plt.legend()
    now_utc = datetime.now(timezone.utc)
    plt.savefig(str(now_utc)+'.pdf')


    shutil.rmtree("single-coil-amp%.6f" % a)
    return np.random.uniform()
#     return N

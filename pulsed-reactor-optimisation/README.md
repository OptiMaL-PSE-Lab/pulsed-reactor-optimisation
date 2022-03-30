<a href="https://www.imperial.ac.uk/optimisation-and-machine-learning-for-process-engineering/about-us/">
<img src="https://avatars.githubusercontent.com/u/81195336?s=200&v=4" alt="Optimal PSE logo" title="OptimalPSE" align="right" height="150" />
</a>

# Pulsed Reactor Optimisation  

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains code to optimise the geometric parameters and operating conditions of a computational fluid dynamics model of a pulsed flow helical tube reactor. 

As the CFD model is computationally expensive to evaluate, this is done through Bayesian optimisation.

## Requirements 

- A working Conda environment.

This repository has been tested on a Unix-based operating system. 

## Instructions 

Build the Conda environment from the ```environment.yml``` file as follows:
```
$ conda env create -f environment.yml
```
Then activate the environment:

```
$ conda activate pulsed-reactor-optimisation_env
```

Running ```pulsed-reactor-optimisation/main.py``` will perform the Bayesian optimisation routine, initialising GPs based on the existing data within ```data/```.

```
$ python pulsed-reactor-optimisation/main.py
```

Author - Tom Savage, 05/09/2021


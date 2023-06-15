<a>
<img src="images/inversion_demo.png" alt="inversion demonstration" title="inversion demonstration" align="center" />
</a>

# Multi-fidelity Bayesian Optimisation for Simulated Chemical Reactors

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


## Outline

```pulsed_flow_helical_tube``` contains the code to optimise helical-tube reactor geometry and operating conditions.

```mesh_generation``` contains mesh generation code. Helical tube reactors are meshed in Python using [```classy_blocks```](https://github.com/damogranlabs/classy_blocks). Meshes are generated based on parameterisation variables, as well as fidelities. 

```example/``` contains an example on how to use the repository. Note the function used is relatively naive, so results should be taken with a pinch of salt. Importantly, the objective is **maximised**. 

## Instructions for Simulated Chemical Reactors

- Install the classy_blocks library as a submodule in the mesh_generation folder
```
$ cd mesh_generation
$ git submodule add git@github.com:damogranlabs/classy_blocks.git
```

- Create and activate the Anaconda environment
```
$ conda env create -f environment.yml
$ conda activate mf_design_env
```


### Requirements
- Strongly advise using Linux.
- OpenFOAM V1906
- Anaconda
- [Swak4FOAM download](https://openfoamwiki.net/index.php/Installation/swak4Foam/Downloading) (development version)
- [Swak4FOAM install](https://openfoamwiki.net/index.php/Installation/swak4Foam)


## References 

```

@misc{darts,
  doi = {10.48550/ARXIV.2305.00710},
  url = {https://arxiv.org/abs/2305.00710},
  author = {Savage,  Tom and Basha,  Nausheen and McDonough,  Jonathan and Matar,  Omar K and Chanona,  Ehecatl Antonio del Rio},
  keywords = {Computational Engineering,  Finance,  and Science (cs.CE),  Optimization and Control (math.OC),  FOS: Computer and information sciences,  FOS: Computer and information sciences,  FOS: Mathematics,  FOS: Mathematics},
  title = {Multi-Fidelity Data-Driven Design and Analysis of Reactor and Tube Simulations},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}



@misc{dgp_based_mfbo,
  doi = {10.48550/ARXIV.2210.17213},
  url = {https://arxiv.org/abs/2210.17213},
  author = {Savage,  Tom and Basha,  Nausheen and Matar,  Omar and Chanona,  Ehecatl Antonio Del-Rio},
  title = {Deep Gaussian Process-based Multi-fidelity Bayesian Optimization for Simulated Chemical Reactors},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```

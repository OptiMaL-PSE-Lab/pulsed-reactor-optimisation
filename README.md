<a>
<img src="images/inversion_demo.png" alt="inversion demonstration" title="inversion demonstration" align="center" />
</a>

# Multi-fidelity Bayesian Optimisation for Simulated Chemical Reactors

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


## Outline

```pulsed_flow_helical_tube``` contains the code to optimise helical-tube reactor geometry and operating conditions.

```mesh_generation``` contains mesh generation code. Helical tube reactors are meshed in Python using [```classy_blocks```](https://github.com/damogranlabs/classy_blocks). Meshes are generated based on parameterisation variables, as well as fidelities. 

```example/``` contains an example on how to use the repository. Note the function used is relatively naive, so results should be taken with a pinch of salt. Importantly, the objective is **maximised**. 


## References 

```
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
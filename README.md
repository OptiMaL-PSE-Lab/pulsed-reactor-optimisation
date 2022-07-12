<a href="https://www.imperial.ac.uk/optimisation-and-machine-learning-for-process-engineering/about-us/">
<img src="https://avatars.githubusercontent.com/u/81195336?s=200&v=4" alt="Optimal PSE logo" title="OptimalPSE" align="right" height="150" />
</a>

# CFD Optimisation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 



This repository contains a number of projects concerned with the optimisation of operating conditions, and geometry of coiled tube reactors. 

<a>
<img src="https://github.com/OptiMaL-PSE-Lab/pulsed-reactor-optimisation/blob/96d12e1a3f5d7a5a943b59d68e402826464d77f8/mesh_generation/output_images/pre_render_basic.png" alt="pre_render" title="pre_render" align="left" height="200" />
</a>



- ```pulsed-reactor-optimisation``` aims to optimise the geometric parameters and operating conditions of a pulsed flow helical tube reactor via CFD. More information can be found within the project's README. Importantly the optimisation and CFD is decoupled and must be done by a human.

<a>
<img src="https://github.com/OptiMaL-PSE-Lab/pulsed-reactor-optimisation/blob/96d12e1a3f5d7a5a943b59d68e402826464d77f8/mesh_generation/output_images/coil_basic_render.png" alt="pre_render" title="pre_render" align="right" height="200" />
</a>

- ```mesh_generation``` contains tools for parametrically creating meshes of coiled tubes, general helices, and other reactors of interest.

- ```simulation-integration``` provides an example of how OpenFOAM and Python can be linked using PyFOAM for the purposes of optimisation. With a case study regarding the optimisation of coiled tube reactor operating conditions. The CFD and optimisation are completely coupled here, allowing for automated evaluation on a high performance cluster. 



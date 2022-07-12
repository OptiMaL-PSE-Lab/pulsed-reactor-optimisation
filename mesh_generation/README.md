<a href="https://www.imperial.ac.uk/optimisation-and-machine-learning-for-process-engineering/about-us/">
<img src="https://avatars.githubusercontent.com/u/81195336?s=200&v=4" alt="Optimal PSE logo" title="OptimalPSE" align="right" height="150" />
</a>

# Mesh Generation 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 



This folder contains Python tools to create a number of different OpenFOAM meshes from parameters via the Classy Blocks library. 

<a>
<img src="https://github.com/OptiMaL-PSE-Lab/pulsed-reactor-optimisation/blob/7064fee30d2c8a90b7935f14332aea63238e5af0/mesh_generation/pre_render.png" alt="pre_render" title="pre_render" align="left" height="250" />
</a>

```coil_basic.py``` contains a function that given a coil radius, pipe radius, number of coils, and overall coil height, will return a blockMesh folder containing the mesh. An O-grid topology is used to create the coil sections. 

```coil_cylindrical.py``` contains a function that given cylindrical coordinates will return a blockMesh folder containing the mesh for a general helix.
This function interpolates between cylindrical coordinates by a given factor to produce the complete set.

A preview of the mesh named ```pre_render.png```is plotted using Matplotlib as the mesh is being constructed for debugging purposes.




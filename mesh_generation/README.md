<a href="https://www.imperial.ac.uk/optimisation-and-machine-learning-for-process-engineering/about-us/">
<img src="https://avatars.githubusercontent.com/u/81195336?s=200&v=4" alt="Optimal PSE logo" title="OptimalPSE" align="right" height="150" />
</a>

# Mesh Generation 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 



This folder contains Python tools to create a number of different OpenFOAM meshes from parameters, coordinates or otherwise via the Classy Blocks library. 

<a>
<img src="https://github.com/OptiMaL-PSE-Lab/pulsed-reactor-optimisation/blob/96d12e1a3f5d7a5a943b59d68e402826464d77f8/mesh_generation/output_images/pre_render_basic.png" alt="pre_render" title="pre_render" align="left" height="250" />
</a>

<a>
<img src="https://github.com/OptiMaL-PSE-Lab/pulsed-reactor-optimisation/blob/96d12e1a3f5d7a5a943b59d68e402826464d77f8/mesh_generation/output_images/coil_basic_render.png" alt="pre_render" title="pre_render" align="right" height="250" />
</a>


```coil_basic.py``` contains a function that given a coil radius, pipe radius, number of coils, and overall coil height, will return a blockMesh folder containing the mesh. An O-grid topology is used to create the coil sections. 

```coil_cylindrical.py``` contains a function that given cylindrical coordinates, and tube radius values will return a blockMesh folder containing the mesh for a general helix.
This function interpolates between given values by a given factor to produce the complete set.

A preview of the mesh named ```pre_render.png```is plotted using Matplotlib as the mesh is being constructed for debugging purposes.







## Notes & Requirements

```input```, ```output```, and ```walls``` patches are defined for OpenFOAM.

[classy_blocks](https://github.com/damogranlabs/classy_blocks) should be installed (added as a submodule) within **this** folder. This can be achieved by: 

- Deleting the existing (empty) folder
```
rm -r classy_blocks
```
- Removing the git link
```
git rm -r classy_blocks
```
- Adding classy_blocks as a submodule
```
git submodule add https://github.com/damogranlabs/classy_blocks
```

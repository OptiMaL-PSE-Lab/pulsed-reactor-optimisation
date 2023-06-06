#!/bin/bash
#PBS -N fidelity_axial_radial_maxCo_deltaT
#PBS -j oe
#PBS -o usable_fidelities/axial_radial_maxCo_deltaT/logs.out
#PBS -e usable_fidelities/axial_radial_maxCo_deltaT/logs.err
#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -lwalltime=01:00:00

echo $cpus
module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B usable_fidelities/fidelity_study.py 1.5 1.5 2 $cpus axial_radial_maxCo_deltaT
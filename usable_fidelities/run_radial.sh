#!/bin/bash
#PBS -N fidelity_radial
#PBS -j oe
#PBS -o what_is_a_fidelity/radial/logs.out
#PBS -e what_is_a_fidelity/radial/logs.err
#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -lwalltime=01:00:00

echo $cpus
module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B what_is_a_fidelity/fidelity_study.py 1.5 1.5 2 $cpus radial
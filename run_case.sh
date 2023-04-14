#!/bin/bash
#PBS -N individual_case
#PBS -j oe
#PBS -o run_case.out
#PBS -e run_case.out
#PBS -lselect=1:ncpus=64:mem=64gb
#PBS -lwalltime=8:00:00

echo $cpus
module load anaconda3/personal
module load openfoam/1906
module load intel-suite


cd $PBS_O_WORKDIR
cd parameterisation_study/cross_section/simulations/2023_04_14_02_54_39

blockMesh
checkMesh -constant
pimpleFoam
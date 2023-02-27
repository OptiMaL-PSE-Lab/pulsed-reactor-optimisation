#!/bin/bash
#PBS -N cross_section
#PBS -j oe
#PBS -o parameterisation_study/cross_section/logs.out
#PBS -e parameterisation_study/cross_section/logs.err
#PBS -lselect=16:ncpus=32:mem=64gb
#PBS -lwalltime=72:00:00

module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B parameterisation_study/cross_section.py parameterisation_study/cross_section/data.json 1.5 2.5 2

#!/bin/bash
#PBS -N run_standard
#PBS -j oe
#PBS -o crlogs.out
#PBS -e crlogs.err
#PBS -lselect=1:ncpus=48:mem=64gb
#PBS -lwalltime=71:00:00

module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B parameterisation_study/run_standard.py $cpus cylinder

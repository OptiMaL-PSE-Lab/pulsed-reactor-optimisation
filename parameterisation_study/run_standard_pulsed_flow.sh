#!/bin/bash
#PBS -N standard_pulsed_flow
#PBS -j oe
#PBS -o parameterisation_study/standard_pulsed_flow/logs.out
#PBS -e parameterisation_study/standard_pulsed_flow/logs.err
#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -lwalltime=01:00:00

echo $cpus
module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B parameterisation_study/standard_pulsed_flow.py parameterisation_study/standard_pulsed_flow/data.json 1.5 1.5 2 $cpus
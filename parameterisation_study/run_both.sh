#!/bin/bash
#PBS -N both
#PBS -j oe
#PBS -o parameterisation_study/both/logs.out
#PBS -e parameterisation_study/both/logs.err
#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -lwalltime=01:00:00

echo $cpus
module load anaconda3/personal
module load openfoam/1906
module load intel-suite
module load cuda/11.4.2
module load cudnn/8.2.4

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B parameterisation_study/both.py parameterisation_study/both/data.json 1.5 1.5 2 $cpus
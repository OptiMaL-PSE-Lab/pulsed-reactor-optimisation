#!/bin/bash
#PBS -N cylindrical_discrepancy
#PBS -j oe
#PBS -o parameterisation_study/cylindrical_discrepancy/logs.out
#PBS -e parameterisation_study/cylindrical_discrepancy/logs.err
#PBS -lselect=1:ncpus=48:mem=48gb
#PBS -lwalltime=72:00:00

module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B parameterisation_study/cylindrical_discrepancy.py parameterisation_study/cylindrical_discrepancy/data.json 1.5 2.5 2

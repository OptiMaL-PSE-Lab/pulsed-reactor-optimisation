#!/bin/bash
#PBS -N sym_dat_high
#PBS -j oe
#PBS -o symbolic_mf_data_generation/high/logs.out
#PBS -e symbolic_mf_data_generation/high/logs.err
#PBS -lselect=1:ncpus=48:mem=64gb
#PBS -lwalltime=72:00:00

module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B symbolic_mf_data_generation/generate_mf_data.py symbolic_mf_data_generation/high/data.json $cpus 50 5 20

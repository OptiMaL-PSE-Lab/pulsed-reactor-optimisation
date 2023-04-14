#!/bin/bash
#PBS -N sym_dat
#PBS -j oe
#PBS -o symbolic_mf_data_generation/low/logs.out
#PBS -e symbolic_mf_data_generation/low/logs.err
#PBS -lselect=1:ncpus=48:mem=64gb
#PBS -lwalltime=72:00:00

module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B symbolic_mf_data_generation/generate_mf_data.py symbolic_mf_data_generation/low/data.json $cpus 30 3 80

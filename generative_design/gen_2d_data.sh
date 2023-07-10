#!/bin/bash
#PBS -N 2d_gen
#PBS -j oe
#PBS -o generative_design/logs.out
#PBS -e generative_design/logs.out
#PBS -lselect=1:ncpus=64:mem=64gb
#PBS -lwalltime=24:00:00

module load anaconda3/personal

cd $PBS_O_WORKDIR
python3 generative_design/2d_reactor_synthesis.py
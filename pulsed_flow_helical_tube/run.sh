#!/bin/bash
#PBS -N pulsed_flow_helical_tube
#PBS -j oe
#PBS -o pulsed_flow_helical_tube/logs.out
#PBS -e pulsed_flow_helical_tube/logs.err
#PBS -lselect=1:ncpus=48:mem=64gb
#PBS -lwalltime=72:00:00

module load anaconda3/personal
module load openfoam/1906
module load intel-suite

cd $PBS_O_WORKDIR
source activate mf_design_env
python3 -B pulsed_flow_helical_tube/pulsed_flow_helical_tube.py

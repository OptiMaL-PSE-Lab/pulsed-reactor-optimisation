#!/bin/bash
qsub -l select=1:ncpus=$1:mem=64gb:ngpus=1:gpu_type=RTX6000 -l walltime=$2:00:00 -v cpus=$1 $3
#!/bin/bash
qsub -l select=1:ncpus=$1:mem=64gb -l walltime=$2:00:00 -v cpus=$1 $3
#!/bin/bash

# Run with: "qsub -I scriptname".

#PBS -l select=5:ncpus=5:mem=20gb

# Maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually).
#PBS -l walltime=01:00:00

# Set the execution queue.
#PBS -q common_cpuQ

python /home/federico.rubbi/fedff/code/fedff2.3/fedff.py
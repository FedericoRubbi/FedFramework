#!/bin/bash

# Run with: "qsub -I scriptname".

#PBS -l select=12:ncpus=10:mem=25gb

# Maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually).
#PBS -l walltime=48:00:00

# Set the execution queue.
#PBS -q common_cpuQ

# Redirect output to log file.
# -e /home/federico.rubbi/project/FedFF/log/job.log

python3 /home/federico.rubbi/project/FedFF/fedff.py

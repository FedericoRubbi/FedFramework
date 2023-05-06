#!/bin/bash

# Run with: "qsub -I scriptname".

#PBS -l select=12:ncpus=10:mem=25gb

# Maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually).
#PBS -l walltime=48:00:00

# Set the execution queue.
#PBS -q common_cpuQ

# Redirect output to log file.
#PBS -e /home/federico.rubbi/project/FedFF/log/job.log
module load python-3.8.13
module load cuda-11.1
source /home/federico.rubbi/flenv/bin/activate
#export PYTHONPATH=/home/federico.rubbi/flenv/lib/python3.8/site-packages:$PYTHONPATH
python --version
python /home/federico.rubbi/project/FedFF/fedff.py

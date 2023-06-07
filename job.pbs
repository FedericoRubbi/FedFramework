#!/bin/bash

# Run with: "qsub -I scriptname".

#PBS -l select=13:ncpus=10:mem=20gb

# Maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually).
#PBS -l walltime=47:00:00

# Set the execution queue.
#PBS -q common_cpuQ

# Redirect output to log file.
#PBS -e /home/federico.rubbi/project/FedFF/log/job.log
module load python-3.8.13
module load cuda-11.1
source /home/federico.rubbi/flenv/bin/activate
python --version
JOBPATH="/home/federico.rubbi/project/FedFF/main.py"
DATE=$(date -I)
echo "Date: $DATE"
echo "Starting job"
python $JOBPATH
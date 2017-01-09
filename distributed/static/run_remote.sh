#!/bin/sh

#parse arguments

#the machine that should be ran on
machine=$1
#the cluste file
clusterfile=$2
#the name of the job
job_name=$3
#the task index
task_index=$4
#the experiments directory
expdir=$5

current_dir=$(pwd)

#ssh to the machine and run the training script
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $machine \
"cd $current_dir && \
python train.py \
--clusterfile=$clusterfile \
--job_name=$job_name \
--task_index=$task_index \
--expdir=$expdir"

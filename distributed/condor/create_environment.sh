#!/bin/sh

#create the necesary environment variables
source ~spch/soft/sgu21/srclogin/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/users/spraak/vrenkens/lib:/users/spraak/vrenkens/lib64:/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/users/spraak/vrenkens/.local/lib/python2.7/site-packages:$PYTHONPATH
export NVIDIA_CUDA=/usr/local/cuda/
export PATH=$PATH:/usr/local/cuda/bin:/usr/bin

#run the original
$@

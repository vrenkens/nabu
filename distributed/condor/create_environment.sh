#!/bin/sh

#create the necesary environment variables
source ~spch/soft/sgu21/srclogin/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:/users/spraak/vrenkens/lib:/users/spraak/vrenkens/lib64:$LD_LIBRARY_PATH
if [ -d "/usr/local/cuda-8.0/lib64" ]
then
  echo "using local cuda"
  export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
else
  echo "using nfs cuda"
  export LD_LIBRARY_PATH=/users/spraak/vrenkens/cuda-8.0/lib64:$LD_LIBRARY_PATH
fi
export PYTHONPATH=/users/spraak/vrenkens/.local/lib/python2.7/site-packages:$PYTHONPATH
export NVIDIA_CUDA=/usr/local/cuda-8.0/
export PATH=$PATH:/usr/local/cuda-8.0/bin:/usr/bin


#run the original
$@

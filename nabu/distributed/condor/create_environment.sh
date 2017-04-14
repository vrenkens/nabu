#!/bin/sh

#create the necesary environment variables
source ~spch/soft/sgu21/srclogin/.bashrc

if [ -d "/usr/local/cuda-8.0/lib64" ]
then
  echo "using local cuda"
  export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
else
  echo "using nfs cuda"
  export LD_LIBRARY_PATH=/users/spraak/vrenkens/cuda-8.0/lib64:$LD_LIBRARY_PATH
fi

#copy the ssh binary to enable ssh tunnels
cp /usr/bin/ssh /tmp

#run the original
$@

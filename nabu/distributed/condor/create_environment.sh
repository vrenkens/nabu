#!/bin/sh

#create the necesary environment variables
source ~/.bashrc

#copy the ssh binary to enable ssh tunnels
cp /usr/bin/ssh /tmp

#run the original
$@

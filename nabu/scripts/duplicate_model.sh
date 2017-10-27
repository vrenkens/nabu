#!/bin/sh

#this file will duplicate the model to several subfolders to allow seperate
#testing on multiple datasets
#
#usage: duplicate_model.sh <expdir> <target> <duplicates>
#   expdir: points to the expdir containng the model to be duplicated
#   target: the target directory

expdir=$(readlink -m $1)
target=$(readlink -m $2)

mkdir -p $target
ln -s $expdir/model $target/model

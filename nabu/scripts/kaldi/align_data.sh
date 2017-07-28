#!/bin/bash

#process inputs

#the directory containing the data to align
datadir=$1
#the directory containing the language model
langdir=$2
#the target directory of the train_gmm script
traindir=$3
#the target directory
targetdir=$4
#location of the kaldi root directory
kaldi=$5

nj=4

cd $kaldi/egs/wsj/s5

mkdir -p $targetdir
mkdir -p $targetdir/mfcc

#compute the mfcc features
echo "----computing MFCC features----"
steps/make_mfcc.sh --nj 20 \
 $datadir $targetdir/make_mfcc $targetdir/mfcc || exit 1;
steps/compute_cmvn_stats.sh $datadir $targetdir/mfcc $targetdir/mfcc || exit 1;

mkdir -p $targetdir/tri_ali

echo "----aligning the data----"

#align the training data with the triphone gmm
steps/align_si.sh --nj $nj --cmd "run.pl" $datadir $langdir $traindir/tri $targetdir/tri_ali

echo "----converting alignments to pdfs----"

#convert the alignments to pdfs
> $targetdir/pdfs
kaldisrc=$kaldi/src
kaldisrc=$(readlink -m $kaldisrc)
for i in $(seq 1 $nj); do
  gunzip -c $targetdir/tri_ali/ali.$i.gz | $kaldisrc/bin/ali-to-pdf $targetdir/tri_ali/final.mdl ark:- ark,t:- >> $targetdir/pdfs
done

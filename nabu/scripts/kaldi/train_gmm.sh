#!/bin/bash

#process inputs

#the directory containing the data to align
datadir=$1
#the directory containing the language model
langdir=$2
#the language directory for the model used for testing
testlang=$3
#the directory where everything will be stored
traindir=$4
#location of the kaldi root directory
kaldi=$5

nj=4

cd $kaldi/egs/wsj/s5

mkdir -p $traindir
mkdir -p $traindir/mfcc

#compute the mfcc features
echo "----computing MFCC features----"
steps/make_mfcc.sh --nj 20 \
 $datadir $traindir/make_mfcc $traindir/mfcc || exit 1;
steps/compute_cmvn_stats.sh $datadir $traindir/mfcc $traindir/mfcc || exit 1;

mkdir -p $traindir/mono

echo "----training monophone gmm----"

#train the monophone gmm
steps/train_mono.sh --nj $nj --cmd "run.pl" $datadir $langdir $traindir/mono

mkdir -p $traindir/mono_ali

echo "----aligning the data----"

#align the training data with the monophone gmm
steps/align_si.sh --nj $nj --cmd "run.pl" $datadir $langdir $traindir/mono $traindir/mono_ali

mkdir -p $traindir/tri

echo "----training triphone gmm----"

#train the triphone gmm
steps/train_deltas.sh --cmd "run.pl" --cluster-thresh 100 3100 50000 $datadir $langdir $traindir/mono_ali $traindir/tri

mkdir -p $traindir/tri_ali

echo "----aligning the data----"

#align the training data with the triphone gmm
steps/align_si.sh --nj $nj --cmd "run.pl" $datadir $langdir $traindir/tri $traindir/tri_ali

echo "----converting alignments to pdfs----"
kaldisrc=$kaldi/src
kaldisrc=$(readlink -m $kaldisrc)
#convert the alignments to pdfs
> $traindir/pdfs
for i in $(seq 1 $nj); do
  gunzip -c $traindir/tri_ali/ali.$i.gz | $kaldisrc/bin/ali-to-pdf $traindir/tri_ali/final.mdl ark:- ark,t:- >> $traindir/pdfs
done

#build the decoding graphs
utils/mkgraph.sh $testlang $traindir/tri $traindir/graph

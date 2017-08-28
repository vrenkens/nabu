#!/bin/bash

nj=4 # number of decoding jobs.  If --transform-dir set, must match that number!
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
cmd=run.pl
beam=15.0
max_active=7000
min_active=200
lattice_beam=8.0 # Beam we use in lattice generation.
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
minimize=false


#the directory where the data is stored
datadir=$1
#the directory used in train gmm
traindir=$2
#the directory containing the log likelihoods
loglikes=$3
#the directory where the outputs should be stored
outputs=$4
#location of the kaldi root directory
kaldi=$5

cd $kaldi/egs/wsj/s5

#create links of data files
ln -s $datadir/utt2spk $loglikes/utt2spk
ln -s $datadir/spk2utt $loglikes/spk2utt

[ -f ./path.sh ] && . ./path.sh; # source the path.

#split the data
sdata=$loglikes/split$nj;
mkdir -p $outputs/log
[[ -d $sdata && $loglikes/feats.scp -ot $sdata ]] || split_data.sh $loglikes $nj || exit 1;
echo $nj > $outputs/num_jobs

#create the reader for the log likelihoods
loglikes="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"

model=$traindir/tri_ali/final.mdl
graphdir=$traindir/graph

#generate the latices
$cmd JOB=1:$nj $outputs/log/decode.JOB.log \
    latgen-faster-mapped \
     --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam \
     --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
     --word-symbol-table=$graphdir/words.txt "$model" \
     $graphdir/HCLG.fst "$loglikes" "ark:|gzip -c > $outputs/lat.JOB.gz" || exit 1;

#rescore at several acoutic weights to get final output
local/score.sh $datadir $graphdir $outputs

cat $outputs/scoring_kaldi/best_wer

[![Build Status](https://travis-ci.org/kaldi-asr/kaldi.svg?branch=master)]
(https://travis-ci.org/kaldi-asr/kaldi)

Kaldi with TensorFlow Neural Net
================================

Installation
--------------------------

- Download and install [TensorFlow](https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#download-and-setup).
- Download and install [Kaldi](https://github.com/kaldi-asr/kaldi)
- Modify the config/config_*.cfg for your setup, specifically the directories

Code overview
--------------------------

main.py: Goes through the neural net training procedure, look at the config files in the config directory to modify the settings
- Compute the features of training and testing set
- Train the monophone GMM with kaldi and get alignments
- Train the triphone GMM with kaldi and get alignments
- Train the neural net with TensorFlow with the alignments as targets
- Get the state likelihoods of the testing set using the neural net
- Decode the testing set with Kaldi using the state likelihoods and report the results

feat.py: Does feature computation currently supports:
- mfcc
- fbank

prepare_data.py: data prep functionality
- compute the features for al the utterances
- shuffle the examples for mini-batch training

kaldi_io.py: functionality to interface with kaldi
- read alignments
- read scp files
- create dummy neural net for decoding
- read and write kaldi ark format

nnet.py: neural network functionality, following operations are possible:
- init: initialize the neural net with greedy layer-by-layer initialisation
- train: train the neural net in mini-batches with SGD (with or without momentum)
- prior: compute the state prior probabilities
- decode: compute the state pseudo-likelihoods



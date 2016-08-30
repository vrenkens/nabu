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
- Compute the features of training and testing set for GMM and DNN
- Train the monophone GMM with kaldi and get alignments
- Train the triphone GMM with kaldi and get alignments
- train the LDA+MLLT GMM with kaldi and get alignments
- Train the neural net with TensorFlow with the alignments as targets
- Get the state pseudo-likelihoods of the testing set using the neural net
- Decode the testing set with Kaldi using the state pseudo-likelihoods and report the results

features/feat.py: Does feature computation currently supports:
- mfcc
- fbank
- ssc

features/prepare_data.py: data prep functionality
- compute the features for all the utterances
- compute mean and variance statistics for normalisation
- shuffle the examples for mini-batch training

io/ark.py: read and write ark format
- defines a reader class for ark format
- defines a writer class for ark format

io/batdispenser.py: reading and formatting features 
- defines a class that can read features and do some processing like splicing and cmvn
- defines a class that can create batches of data 

neural networks/nnet.py: neural network class for a kaldi-style neural network 
- train: train the neural net 
- decode: compute pseudo-likelihood

neural networks/nnetgraph.py: creating tensorflow graph structures
- defines a abstract class for a neural network graph
- defines a class for a deep neural network, inherits from the neural network graph class
- defines a class for the training environment for a neural network
- defines a class for the decoding environment of a neural network

neural networks/nnetlayer.py: layers for a neural network
- defines a feed forward fully connected layer



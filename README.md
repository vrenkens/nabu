#Nabu

Please find the documentation page [here](http://vrenkens.github.io/nabu)

##Table of content
- [About](#about)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Database preperation](#database-preperation)
    - [ASR Database preparation](#asr-database-preparation)
    - [LM Database preparation](#lm-database-preparation)
  - [Data preperation](#database-preperation)
  - [Training a model](#training-a-model)
    - [Configuration](#configuration)
    - [Running the training script](#running-the-training-script)
    - [Visualization](#visualization)
  - [Testing a model](#testing-a-model)
- [Design](#design)
  - [Designing a model](#designing-a-model)
    - [Creating the classifier](#creating-the-classifier)
    - [Classifier configuration file](#classifier-configuration-file)
  - [Designing features](#designing-features)
    - [Creating the feature computer](#creating-the-feature-computer)
    - [Feature configuration file](#feature-configuration-file)
  - [Designing a trainer](#designing-a-trainer)
    - [Creating the trainer](#creating-the-trainer)
    - [Trainer configuration file](#trainer-configuration-file)
  - [Designing a decoder](#designing-a-decoder)
    - [Creating the decoder](#creating-the-decoder)
    - [Decoder configuration file](#decoder-configuration-file)
- [Distributed training](#distributed-training)
  - [Non-distributed](#non-distributed)
  - [Local](#local)
  - [Static](#static)
  - [Condor-local](#condor-local)
  - [Condor](#condor)



##About

Nabu is a toolbox for designing, training and using end-to-end neural network
systems for Automatic Speech Recognition. Nabu is built on top of TensorFlow.
Some model architectures have been implemented in Nabu e.g.:

- Deep Bidirectional LSTM with CTC
- Attention based encoder-decoder networks

Nabu's design is focused on adaptability, so users can easily design there own
models and methodologies.

##Dependencies

- [TensorFlow](https://www.tensorflow.org).

##Usage

###Database preperation

####ASR Database preparation

Nabu's data format for asr is the same as the [Kaldi](http://kaldi-asr.org/)
toolkit. To prepare a database for training you should run Kaldi's data
preparation scripts, these scripts can be found on the Kaldi Github.
You can find more information on the Kaldi data preparation
[here](http://kaldi-asr.org/doc/data_prep.html).

The resulting data directories should contain:

- wav.scp: a text file with a line per utterance, this line should contain the
ID of the utterance and either a wav filename or a command to read the audio
file and pipe the results. Examples:
  - utt1 /path/to/file.wav
  - utt1 sph2pipe -f wav /path/to/file.wav |
- a text file (name does not matter) containing the reference transcription for
all utterances. Example:
  - utt1 this is a reference transcription
- spk2utt: a file containing a mapping from speaker to utterance, every line
contains the utterances for a speaker (space separated). If no speaker
information is given just put all utterances on a single line. Example:
  - speaker1 utt1 utt2 utt3
- utt2spk: the reverse mapping for spk2utt one line per utterance containing
the utterance ID and the speaker ID. Example:
  - utt1 speaker1

Finally, you should create a config file for your database in
config/asr_databases/. You can base your config on
config/asr_databases/template.cfg:

- _data fields should point to the directories created in the database
preparations
- _dir fields should contain writable directories where your features
should be stored
- Text fields should point to the transcription text files created in the
database preparation.
- The normalizer field should be the name of the target normalizer (defined in
the normalizer factory). Look at the
[Target normalizer section](#target-normalizer).

####LM Database preparation

For training a language model the only thing that is required is a (or multiple)
text file(s) where each line in the file is a seperate sentence.

Finally, you should create a config file for your database in
config/lm_databases/. You can base your config on
config/lm_databases/template.cfg:

- _data fields should point to the text files created in the database
preparations, this is a space seperated list of files
- _dir fields should contain writable directories where data can be stored
- The normalizer field should be the name of the target normalizer (defined in
the normalizer factory). Look at the
[Target normalizer section](#target-normalizer).

####Target normalizer

For every database you should create a target normalizer in
nabu/processing/target_normalizers. The Normalizer class is defined in
nabu/processing/target_normalizers/normalizer.py. To create the normalizer you
should inhererit from the Normalizer class and overwrite the \_\_call\_\_ method
and the _create_alphabet method.

The _create_alphabet method creates the alphabet of targets. It returns a list
of target strings. spaces are not allowed in the target strings.

The \_\_call\_\_ method takes a transcription string from the text file  as input
and returns the normalized transcription as a string. The normalized
transcription should contain only targets from the alphabet seperated by spaces.
An example normalized transcription:

> i &lt;space> a m &lt;space> a &lt;space> t r a n s c r i p t i o n

An example normalizer can be found in
nabu/processing/target_normalizers/aurora4.py. Once you've created your
normalizer you should import it in
nabu/processing/target_normalizers/\_\_init\_\_.py and add it to the factory
method in nabu/processing/target_normalizers/normalizer_factory.py with a name
that matches the name in the database config file.

###Data preperation

To to the asr or lm data preperation (feature computation, text normalization
etc.) you can use the asr_dataprep.py and lm_dataprep.py scripts. You should
first make sure that the database_cfg_file variable at the top of the scripts
point to the config that you created in the
[database preperation](#database-preperation). For the asr database preperation
you should also point the feat_cfg_file variable to a feature config file in
config/features/.
To design your own features look into the
[Designing features section](#designing-features).

You can then do the data preperation with

```
python asr_dataprep.py
```
or
```
python lm_dataprep.py
```

###Training a model

To train a pre-designed model (see [Designing a model](#designing-a-model) to
design your own) you can use the run_train.py script.

####Configuration

For configuration you can modify the following config files:

- nabu/config/computing/: The computing mode configuration, this will be
explained more in the [Distributed training section](#distributed-training)
(stick to non-distributed.cfg for now).
- nabu/config/asr/: The asr neural network configuration. The content of this
configuration depends on the type of asr. In this section you can choose
the number of layers, the dimensionality of the layers etc. Look at the
[Designing a model section](#designing-a-model) if you want to design
your own type of asr.
- nabu/config/lm/: The lm neural network configuration. The content of this
configuration depends on the type of lm. In this section you can choose
the number of layers, the dimensionality of the layers etc. Look at the
[Designing a model section](#designing-a-model) if you want to design
your own type of lm.
- nabu/config/trainer/: The trainer configuration, this config contains the type
of trainer and its configuration. Select the type of trainer that is appropriate
for your model (e.g. cross_entropy for encoder-decoder nets).You can also set
the training parameters like learning rate, batch size etc. Look at the
[Designing a trainer section](#designing-a-trainer) if you want to design your
own type of trainer.
- nabu/config/decoder/: this configuration contains the type of decoder that
will be used during validation. Choose the type that is appropriate for your
model (e.g. BeamSearchDecoder for encoder-decoder nets). You can also modify
some decoding parameters like beam width and batch size. Look at the [Designing
a decoder section](#designing-a-decoder) if you want to design your own type of
decoder.

When all configuration files have been edited/created to your liking you should
should modify the _cfg_file variables at the top of the run_train.py script so
they point to the appropriate config files.

####Running the training script

You can then train an asr by running:

```
python run_train.py --expdir=path/to/expdir --type=asr
```

and a lm with

```
python run_train.py --expdir=path/to/expdir --type=lm
```

The expdir argument should point to the directory where you want all your
experiment files to be written (model checkpoints, config files, event files
etc).

If you have a validation set, training will start by measuring the initial
performance. Next, the iterative training will start until the required number
of steps have been taken. If your process stopped for some reason, you can
resume training by setting the resume_training field to True in the trainer
config and using the same experiments directory.

####Visualization

During training you can visualize the network, its parameters, performance on
the validation set etc. using
[Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/).
You can start Tensorboard with:

```
tensorboard --logdir=path/to/expdir/logdir
```

or

```
python -m tensorflow.tensorboard --logdir=path/to/expdir/logdir
```

the logdir is created in the expdir.

###Testing a model

To test a trained model you can use the test_asr.py, test_lm.py and test.py
scripts. You can use a different decoder configuration then you used during
training. You should modify the decoder_cfg_file variable at the top of scrip,
so it points to the correct config file. If you want to use the config file that
you used during training set the variable to None.

You can measure the perplexity of a lm on the test set with:

```
python test_lm.py --expdir=path/to/expdir
```

You can test an asr without lm with:

```
python test_asr.py --expdir=path/to/expdir
```

Finally, you can test an asr with lm with:

```
python test.py --asr_expdir=path/to/asr/expdir --lm_expdir=path/to/lm/expdir
```

##Design

###Designing a model

####Creating the classifier

The classifier is the core of the model, and can be either an asr or a language
model. The general Classifier class is defined in
nabu/neuralnetworks/classifiers/classifier.py. To create your own classifier
create a class in nabu/neuralnetworks/classifiers/ that inherits from Classifier
and overwrite the \_\_call\_\_ method. This method takes the following inputs:

- inputs: the inputs to the neural network, this is a
    [batch_size x max_input_length x feature_dim] tensor. If the classifier
    is a language model the feature_dim will be 1
- input_seq_length: The sequence lengths of the input utterances, this
    is a [batch_size] vector
- targets: the targets to the neural network, this is a
    [batch_size x max_output_length x 1] tensor. The targets can be
    used during training
- target_seq_length: The sequence lengths of the target utterances,
    this is a [batch_size] vector
- is_training: whether or not the network is in training mode

The method should return the output logits (probabilities before softmax) and
the output sequence lengths. Some example Classifiers:

- nabu/neuralnetworks/classifiers/las.py: Listen Attend and Spell asr
- nabu/neuralnetworks/classifiers/dblstm.py: Deep Biderictional LSTM asr

Once you've created your classifier you should add it in the factory method in
nabu/neuralnetworks/classifiers/classifier_factory.py (with any name) and you
should import it in nabu/neuralnetworks/classifiers/\_\_init\_\_.py.

####Classifier configuration file

If you've created your classifier you should also create a configuration file
for it in nabu/config/nnet/. The Classifier object will have access to this
configuration as a dictionary of strings in self.conf. You can use this
configuration to set some parameters to your model. As a minimum the
configuration file should contain the classifier field with the name of your
classifier (that you've defined in the factory method). As an example you can
look at other configuration files in nabu/config/nnet/.

###Designing features

####Creating the feature computer

the general FeatureComputer class is defined in
nabu/processing/feature_computers/feature_computer.py. To design your own
feature you should create a feature computer class that inherits from
FeatureComputer and overwrite the comp_feat and the get_dim methods. The
comp_feat method takes the following inputs:

- sig: the audio signal as a 1-D numpy array
- rate: the sampling rate

It returns the computed features as a [seq_length x feature_dim] numpy array.
The get_utt method shpuld return the dimension of the computed features. Some
implemented feature computers:

- nabu/processing/feature_computers/fbank.py
- nabu/processing/feature_computers/mfcc.py

Once your feature computer is created you should add it in the factory method in
nabu/processing/feature_computers/feature_computer_factory.py (with any name)
and you should import it in nabu/processing/feature_computers/\_\_init\_\_.py.

####Feature configuration file

After creating your feature computer you should also create a configuration file
for it in nabu/config/features/. The FeatureComputer object will have access to
this configuration as a dictionary of strings in self.conf. You can use this
configuration to set some parameters for your features. As a minimum it should
contain the following fields:

- name: The name of the feature, this is used for storage and loading
- feature: The feature type. This should be the name that you gave in the
factory method

###Designing a trainer

####Creating the trainer

The general Trainer class is defined in nabu/neuralnetworks/trainers/trainer.py.
To design your own trainer you should create a trainer class in
nabu/neuralnetworks/trainers/ that inherits from Trainer and overwrite the
compute_loss method. This method takes the following inputs:

- targets: a [batch_size, max_target_length] tensor containing the targets
- logits: a [batch_size, max_logit_length, dim] tensor containing the logits
- logit_seq_length: the length of all the logit sequences as a [batch_size]
vector
- target_seq_length: the length of all the target sequences as a [batch_size]
vector

The method should return the loss that you would like to minimize. Some
implemented trainers:

- nabu/neuralnetworks/trainers/cross_entropytrainer.py: minimizes cross-entropy
- nabu/neuralnetworks/trainers/ctctrainer.py: minimizes CTC loss

Once your trainer is created you should add it in the factory method in
nabu/neuralnetworks/trainers/trainer_factory.py (with any name) and you should
import it in nabu/neuralnetworks/trainers/\_\_init\_\_.py.

####Training configuration file

After creating your trainer you should also create a configuration file for it
in nabu/config/trainer/. The Trainer object will have access to this
configuration as a dictionary of strings in self.conf. You can use this
configuration to set some parameters for your trainer. As a minimum it should
contain the fields that are defined in
nabu/config/trainer/cross_entropytrainer.cfg. And you should change the trainer
field to the name that you defined in the factory method.

###Designing a decoder

####Creating the decoder

The general Decoder class is defined in nabu/neuralnetworks/decoders/decoder.py.
To design your own decoder you should create a decoder class in
nabu/neuralnetworks/decoders/ that inherits from Decoder and overwrite the
get_outputs and the score methods. The get_outputs method takes the following
inputs:

- inputs: The inputs to the network as a
[batch_size x max_input_length x input_dim] tensor
- input_seq_length: The sequence length of the inputs as a [batch_size] vector
- classifier: The classifier object that will be used in decoding
- classifier_scope: the scope where the classifier was defined

To make it possible to load or reuse the classifier the classifier should
always be called within the classifier scope. For example if you want to get
the output logits you should do:

```
with tf.variable_scope(classifier_scope):
  logits, lengths = classifier(...)
```

The method should return a list with batch_size elements containing nbest lists.
Each nbest list is a list containing pairs of score and a numpy array with
output labels.

The score method is used to validate the model and takes the following inputs:

- outputs: a dictionary containing nbest lists of decoder outputs
- targets: a dictionary containing the targets

The method should return a score that can be used for validation (e.g. Character
Error Rate). Some example decoders:

- nabu/neuralnetworks/decoders/ctc_decoder.py
- nabu/neuralnetworks/decoders/beam_search_decoder.py

Once your decoder is created you should add it in the factory method in
nabu/neuralnetworks/decoders/decoder_factory.py (with any name) and you should
import it in nabu/neuralnetworks/decoders/\_\_init\_\_.py.

####Decoder configuration file

After creating your decoder you should also create a configuration file for it
in nabu/config/decoder/. The Decoder object will have access to this
configuration as a dictionary of strings in self.conf. You can use this
configuration to set some parameters for your decoder. As a minimum it should
contain the decoder field and you should set it to the name that you defined in
the factory method.

##Distributed training

In distributed training there are 2 jobs. A parameter server (ps) will store the
parameters and share them. A worker computes gradients and submits them to the
parameter servers to update the parameters. The parameter server will run on a
CPU. The workers will run on a GPU if one is available.

For each job you can have multiple instances or tasks, if you have multiple
parameter servers, the parameters will  be divided between them. This can help
if communication is a bottleneck. If  there are many workers the parameter
server may have trouble serving them all in a timely fashion. Multiple workers
can process multiple batches in parallel and update the parameters either
synchronously or asynchronously (depending on the numbatches_to_aggregate field
in the trainer config).

Nabu's distributed computing makes the assumption that the different machines
have access to the same file system. Communication happens through ssh tunnels
to circumvene firewalls on the network ports. SSH Authentication should happen
with an RSA key and not with a password. If one of these things is not possible
for your setup, you can either train non-distributed or you should make sure all
your devices are in a single machine.

There are 5 computing configurations that can be used for training. 2 of the
configurations use the [HTCondor](https://research.cs.wisc.edu/htcondor/)
distributed computing system. If your system is not set up with condor you can
either use the configurations that do not use condor, or implement a
configuration for your setup.

###Non-distributed

To choose this computing configuration you should set the computing_cfg_file to
'nabu/config/computing/non-distributed.cfg' at the top of run_train.py. The
non-distributed computing is the simplest computing form. The training will run
on the machine where it is called from and will run on a single device.

###Local

To choose this computing configuration you should set the computing_cfg_file to
'nabu/config/computing/local.cfg' at the top of run_train.py. Local computing
will run on the machine where it is called from but runs on multiple devices.
You can choose the number of devices in the config file.

###Static

To choose this computing configuration you should set the computing_cfg_file to
'nabu/config/computing/static.cfg' at the top of run_train.py. Static computing
runs on a statically defined cluster over possibly multiple machines. The
cluster should be defined in a cluster file (the clusterfile field in the config
should point to this file). Every line in the cluster file defines a task in the
cluster. Every line contains the job, the machine, the network port for
communication and the GPU index the task should run on (may be empty). All
seperated by commas. An example cluster file:

```
ps,ps1.example.com,1024,
ps,ps2.example.com,1024,
worker,worker1.example.com,1024,0
worker,worker1.example.com,1025,1
worker,worker2.example.com,1024,0
```

###Condor local

To choose this computing configuration you should set the computing_cfg_file to
'nabu/config/computing/condor-local.cfg' at the top of run_train.py. This
configuration is the same as the [Local](#local) configuration, but an
appropriate machine  is found with condor.

###Condor

To choose this computing configuration you should set the computing_cfg_file to
'nabu/config/computing/condor.cfg' at the top of run_train.py. This
configuration is the same as the [Static](#static) configuration, but instead of
using a statically defined cluster, a cluster will be created using Condor.

##Future work

The current future work focusses on incorporating language models into Nabu.

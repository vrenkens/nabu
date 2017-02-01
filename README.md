#Nabu

Please find the documentation page [here](http://vrenkens.github.io/nabu)

##Table of content
- [About](#about)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Data preparation](#data-preparation)
    - [Database preparation](#database-preparation)
    - [Feature computation](#feature-computation)
  - [Training a model](#training-a-model)
    - [Configuration](#configuration)
    - [Running the training script](#running-the-training-script)
    - [Visualization](#visualization)
  - [Testing a model](#testing-a-model)
- [Design](#design)
  - [Designing a model](#designing-a-model)
    - [Creating the classifier](#creating-the-classifier)



##About

Nabu is a toolbox for designing, training and using end-to-end neural network
systems for Automatic Speech Recognition. Nabu is built on top of TensorFlow.
Some model architectures have been implemented in Nabu e.g.:

- Deep Bidirectional LSTM with CTC
- Attention based encodeder-decoder networks

Nabu's design is focussed on adaptability, so users can easily design there own
models and methodologies.

##Dependencies

- [TensorFlow](https://www.tensorflow.org).

##Usage

###Data preparation

####Database preparation

Nabu's data format is the same as the [Kaldi](http://kaldi-asr.org/)
toolkit. To prepare a database for training you should run Kaldi's data
preperation scripts, these scripts can be found on the Kaldi Github.
You can find more information on the Kaldi data preperation
[here](http://kaldi-asr.org/doc/data_prep.html).

The resulting data directories should contain:

- wav.scp: a text file with a line per utterance, this line should contain the
ID of the utterance and either a wav filename or a command to read the audio
file and pipe the results. Examples:
  - utt1 /path/to/file.wav
  - utt1 sph2pipe -f wav /path/to/file.wav |
- a text file containing (name does not matter) the reference transcription for
all utterances. Example:
  - utt1 this is a reference transcription
- spk2utt: a file containing a mapping from speaker to utterance, every line
contains the utterances for a speaker (space seperated). If no speaker
information is given just put all utterances on a single line. Example:
  - speaker1 utt1 utt2 utt3
- utt2spk: the reverse mapping for spk2utt one line per utterance containing
the utterance ID and the speaker ID. Example:
  - utt1 speaker1

Next you should create a database normalizer in processing/target_normalizers/.
This is a function that a transcription and an alphabet as input and returns
the normalized transcription. You can find allready implemented normalizers
in processing/target_normalizers/, you should use these as examples. Once you
implemented your normalizer you should add it in
processing/target_normalizers/normalizer_factory.py in the factory method
(with a name of your choosing). You should also import your new file in
processing/target_normalizers/\__init\__.py.

Next you should create a target coder in processing/target_coders/. A target
coder converts a normalized transcription string into a numpy array of labels.
To create your own coder you should create a class that inherits from
TargetCoder (defined in processing/target_coders/targetcoder.py). You should
then overwrite the create_alphabet method that returns your desired alphabet.
Some coders have been implemented and can be found in processing/target_coders/.
Once you implemented your coder you should add it in
processing/target_coders/coder_factory.py in the factory method
(with a name of your choosing). You should also import your new file in
processing/target_coders/\__init\__.py.

Finally, you should create a config file for your database in config/databases/.
You can base your config on config/databases/template.cfg:

- _data fields should point to the directories created in the database
preperations. The
- _features fields should contain writable directories where your features
should be stored
- text fields should point to the transcription text files created in the
database preperation.
- coder field should be the name of your target coder (defined in the
factory).
- normalizer field should be the name of your target normalizer (defined in the
factory).

####Feature computation

To compute the features you can use the featprep.py script. For the feature
configuration you can modify/create a file in config/features/ or use a
default config file.

In the script you
should modify the database_cfg_file variable to point to the desired database
config file and the feat_cfg_file to point to the desired feature config file.

Then you can compute the features with:

```
python featprep
```

###Training a model

To train a pre-designed model (see [Designing a model](#designing-a-model) to
design your own) you can use the run_train.py script.

####Configuration

For configuration you can modify the folowing config files:

- config/computing/: The computing mode configuration, this will be explained
more in the [Distributed training section](#distributed-training) (stick to
local.cfg for now).
- config/nnet/: The neural network configuration. The content of this
configuration depends on the type of model (). In this section you can choose
the number of layers, the dimensionality of the layers etc.
- config/trainer/: The trainer configuration, this config contains the type of
trainer and its configuration. Select the type of trainer that is appropriate
for your model (e.g. cross_enthropy for encoder-decoder nets).You can also set
the training parameters like learning rate, batch size etc. Look at the
[Designing a trainer section](#designing-a-trainer) if you want to design your
own type of trainer.
- config/decoder/: this configuration contains the type of decoder that will
be used during validation. Choose the type that is appropriate for you model
(e.g. beamsearchdecoder for encoder-decoder nets). You can also modify some
decoding parameters like beam width and batch size.

When all configuration files have been edited/created to your liking you should
should modify the _cfg_file variables at the top of the run_train.py script so
they point to the appropriate config files.

####Running the training script

You can start training with:

```
python run_train.py --expdir=path/to/expdir
```

The expdir argument should point to the directory where you want all your
experiment files to be written (model checkpoints, config files, event files
etc).

If you have a validation set, training will start by measuring the initial
performance. Next, the itarative training will start untill the required number
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

To test a trained model you can use the test.py script. You can use a different
decoder configuration then you used during training. You should modify the
decoder_cfg_file variable at the top of test.py, so it points to the correct
config file. If you want to use the config file that you used during training
set the variable to None.

##Design

###Designing a model

####Creating the classifier

The classifier is the core of the model. The general Classifier class is defined
in neuralnetworks/classifiers/classifier.py. All classifiers inherit from
Classifier and follow the same interface. To create your own classifier define
a class that inherits from Classifier and overwrite the \__call\__ method.
This method takes the folowing inputs:

- inputs: the inputs to the neural network, this is a
    [batch_size x max_input_length x feature_dim] tensor
- input_seq_length: The sequence lengths of the input utterances, this
    is a [batch_size] dimansional vector
- targets: the targets to the neural network, this is a
    [batch_size x max_output_length x 1] tensor. The targets can be
    used during training
- target_seq_length: The sequence lengths of the target utterances,
    this is a [batch_size] dimansional vector
- is_training: whether or not the network is in training mode

The method should return the output logits (probabilities before softmax) and
the output sequence lengths.

The classifier will be called within the trainer and decoder, so the classifier
should not define its own graph. Some example Classifiers:

- neuralnetworks/classifiers/las.py
- neuralnetworks/classifiers/dblstm.py

Once you've created your classifier you should add it in the factory method in
neuralnetworks/classifiers/classifier_factory (with any name) and you should
import it in neuralnetworks/classifiers/\__init\__.py.

####Classifier configuration file

If you've created your classifier you should also create a configuration file
for it in config/nnet/. The Classifier object will have access to this
configuration as a dictionary of strings. You can use this configuration to
set some parameters to your model. As a minimum the configuration file should
contain the classifier field with the name of your classifier (that you've
defined in the factory method). As an example you can look at other
configuration files in config/nnet/.

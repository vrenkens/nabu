# Nabu

Please find the documentation page [here](http://vrenkens.github.io/nabu)

Nabu is an ASR framework for end-to-end networks built on top of TensorFlow.
Nabu's design focusses on adaptibility, making it easy for the designer to
adjust everything from the model structure to the way it is trained.

## Using Nabu

Nabu works in several stages: data prepation, training and finally testing and
decoding. Each of these stages uses a recipe for a specific model and database.
The recipe contains configuration files for the all components and defines all
the necesary parameters for the database and the model. You can find more
information on the components in a recipe [here](config/recipes/README.md).

### Data preperation

In the data preperation stage all the data is prepared (feature computation,
target normalization etc.) for training and testing. Before running the data
preperation you should create a database.conf file in the recipe directory
based on the database.cfg that should already be there, and fill in all the
paths. Should you want to modify parameters in the processors, you can modify
the config files that are pointed to in the database config. You can find more
information about processors [here](nabu/processing/processors/README.md).

You can run the data prepation with:

```
run data --recipe=/path/to/recipe --expdir=/path/to/expdir --computing=<computing>
```

- recipe: points to the directory containing the recipe you
want to prepare the data for.
- expdir: the path to a directory where you can write to. In this directory all
files will be stored, like the configurations and logs
- computing [default: standard]: the distributed computing software you want to
use. One of standard or condor. standard means that no distributed computing
software is used and the job will run on the machine where nabu is called from.
the condor option uses HTCondor. More information can be found
[here](nabu/computing/README.md).

### Training

In the training stage the model will be trained to minimize a loss function.
During training the model can be evaluated to adjust the learning rate if
necessary. Multiple configuration files in the recipe are used during training:

- model.cfg: model parameters
- trainer.cfg: training parameters
- validation_evaluator.cfg: validation parameters

You can find more information about models
[here](nabu/neuralnetworks/models/README.md), about trainers
[here](nabu/neuralnetworks/trainers/README.md) and about evaluators
[here](nabu/neuralnetworks/evaluators/README.md).

You can run the training with:

```
run train --recipe=/path/to/recipe --expdir=/path/to/expdir --mode=<mode> --computing=<computing>
```

The parameters are the same as the data preperation script (see above) with one
extra parameter; mode (default: non_distributed). Mode is the distribution mode.
This should be one of non_distributed, single_machine or multi_machine.
You can find more information about this [here](nabu/computing/README.md)

### Testing

In the testing stage the performance of the model is evaluated on a testing set.
To modify the way the model in is evaluated you can modify the
test_evaluator.cfg file in the recipe dir. You can find more information on
evaluators [here](nabu/neuralnetworks/trainers/README.md).

You can run testing with

```
run test --recipe=/path/to/recipe --expdir=/path/to/expdir --computing=<computing>
```

The parameters for this script are similar to the training script (see above).
You should use the same expdir that you used for training the model.

### Decoding

In the decoding stage the model is used to decode the test set and the resulting
nbest lists are written do disk in the expdir. To modify the way the model is
used for decoding look into the recognizer.cfg file. You can find more
information about decoders [here](nabu/neuralnetworks/decoders/README.md).

You can run decoding with

```
run decode --recipe=/path/to/recipe --expdir=/path/to/expdir --computing=<computing>
```

The parameters for this script are similar to the training script (see above).
You should use the same expdir that you used for training the model.

### Parameter search

You can automatically do a parameter search using Nabu. To do this you should
create a sweep file. A sweep file contain blocks of parameters, each block
will change the parameters in the recipe and run a script. A sweep file
looks like this:

```
experiment name 1
confile1 section option value
confile2 section option value
...

experiment name 2
confile1 section option value
confile2 section option value
...

...
```

For example, if you want to try several number of layers and number of units:

```
4layers_1024units
model.cfg encoder num_layers 4
model.cfg encoder num_units 1024

4layers_1024units
model.cfg encoder num_layers 4
model.cfg encoder num_units 2048

5layers_1024units
model.cfg encoder num_layers 5
model.cfg encoder num_units 1024

5layers_1024units
model.cfg encoder num_layers 5
model.cfg encoder num_units 2048
```

The parameter sweep can then be executed as follows:

```
run sweep --command=<command> --sweep=/path/to/sweepfile --expdir=/path/to/exdir <command option>
```

where command can be any of the commands discussed above.

### Kaldi and Nabu

There are some scripts avaiable to use a Nabu Neural Network in the Kaldi
framework. Kaldi is an ASR toolkit. You can find more information
[here](http://www.kaldi-asr.org/).

Using Kaldi with nabu happens in several steps:
1) Data preperation
2) GMM-HMM training
3) Aligning the data
4) computing the prior
5) Training the Neural Network
6) Decoding and scoring

#### Data preperation

The data preperation is database dependent. Kaldi has many scripts for data
preperation and you should use them.

#### GMM-HMM training

You can train the GMM-HMM model as folows:

```
nabu/scipts/kaldi/train_gmm.sh <datadir> <langdir> <langdir-test>  <traindir> <kaldi>
```

With the folowing arguments:
- datadir: the directory containing the training data (created in data prep)
- langdir: the directory containing the language model for training
(created in data prep)
- langdir-test: the directory containg the language model that should be used
for decoding (created in data prep)
- traindir: The directory where the training files (logs, models, ...) will be
written
- kaldi: the location of your kaldi installation

The script will compute the features, train the GMM-HMM models and align the
training data, so you do not have to do this anymore in the coming step.
The alignments for the training set can be found in &lt;traindir>/pdfs.

#### Aligning the data

The training data has already been aligned in the previous step, but if you want
to align e.g. the validation set you can do that as follows:

```
nabu/scipts/kaldi/align_data.sh <datadir> <langdir> <traindir> <targetdir> <kaldi>
```

the datadir should point to the data you want to align, the traindir should be
the traindir you used in the previous step and the targetdir is the directory
where the alignments will be written. The alignments can be found in
&lt;targetdir>/pdfs

#### Computing the prior

The prior is needed to convert the pdf posteriors to pdf pseudo-likelihoods.
The prior can be computed with:

```
nabu/scipts/kaldi/compute_prior.sh <traindir>
```

traindir should be the same as the traindir in the previous step. the prior can
then be found in numpy format in &lt;traindir>/prior.npy

#### Training the neural net

Training the neural network happens using the Nabu framework. In order to do
this you should create a recipe for doing so (see the section on training).
You can find an example recipe for this in config/recipes/DNN/WSJ. You can
use this recipe, but you should still create the database.conf file based on
the database.cfg file. In your database configuration you should create sections
for the features which is the same as you would do for a normal Nabu neural
network and sections for the alignments. The alignment sections should get the
special type "alignments". A section should look something like this:

```
[trainalignments]
type = alignment
datafiles = <traindir>/pdfs
dir = /path/to/dir
processor_config = path/to/alignment_processor.cfg
```

dir is just the directory where the processed alignments will be written.

The rest of the training procedure is the same as the normal procedure, so
folow the instructions in the sections above.

#### Decoding and scoring

To decode the using the trained system you should first compute the
pseudo-likelihoods as folows:

```
run decode --expdir=<expdir> --recipe=<recipe> ...
```

The pseudo likelihoods can the be found in &lt;expdir>/decode/decoded/alignments.

You can then do the Kaldi decoding and scoring with:

```
nabu/scipts/kaldi/decode.sh <datadir> <traindir> <expdir>/decode/decoded/alignments/feats.scp <outputs> <kaldi>
```

The arguments are similar as the arguments in the script above. The outputs
will be written to the &lt;outputs> folder.

## Designing in Nabu

As mentioned in the beginning Nabu focusses on adaptibility. You can easily
design new models, trainers etc. Most classes used in Nabu have a general class
that defines an interface and common functionality for all children and
a factory that is used to create the necessary class. Look into the respective
README files to see how to implement a new class.

In general, if you want to add your own type of class (like a new model) you
should follow these steps:
- Create a file in the class directory
- Create your child class, let it inherit from the general class and overwrite
the abstract methods
- Add your class to the factory method. In the factory method you give your
class a name, this does not have to be the name of the class. You will use
this name in the configuration file for your model so Nabu knows which class
to use.
- Add your file to the package in \_\_init\_\_.py
- create a configuration file for your class and put it in templates. You
should then add this configuration file in whichever recipe you want to use it
for or create your own recipe using your new class.

# Nabu

Please find the documentation page [here](http://vrenkens.github.io/nabu)

Nabu is an ASR framework for end-to-end networks built on top of TensorFlow.
Nabu's design focusses on adaptibility, making it easy for the designer to
adjust everything from the model structure to the way it is trained.

## Using Nabu

Nabu works in several stages: data prepation, training and finally testing and
decoding. Each of these stages uses a recipe for a specific model and database.
The recipe defines all the necesary parameters for the database and the model.
You can find more information on recipes [here](config/recipes/README.md).

### Data preperation

In the data prepation stage all the data is prepared (feature computation,
target normalization etc.) for training and testing. Before running the data
preperation you should create a database.conf file in the recipe directory
based on the database.cfg that should already be there ,and fill in all the
paths. Should you want to modify parameters in the processors, you can modify
the config files that are pointed to in the database config. You can find more
information about processors [here](nabu/processing/processors/README.md).

You can run the data prepation with:

```
run data --recipe=/path/to/recipe
```

The recipe parameter should point to the directory containing the recipe you
want to prepare the data for.

### Training

In the training stage the model will be trained to minimize a loss function.
During training the model can be evaluated to adjust the learning rate if
necessary. Multile configuration files in the recipe are used during training:

- model.cfg: model parameters
- trainer.cfg: training parameters
- validation_evaluator.cfg: validation parameters

You can find more information about models
[here](nabu/neuralnetworks/models/README.md), about trainers
[here](nabu/neuralnetworks/trainers/README.md) and about evaluators
[here](nabu/neuralnetworks/evaluators/README.md).

You can run the training with:

```
run train --recipe=/path/to/recipe --expdir=/path/to/expdir --mode=<mode>
--computing=<computing>
```

The parameters of this script are the following:

- recipe: path to the recipe configuration directory (like in data prepation)
- expdir: the path to a directory where you can write to. In this directory all
files will be stored, like the configurations, intermediate models, logs etc.
- mode [default: non_distributed]: this is the distribution mode. This should be
one of non_distributed, single_machine or multi_machine. You can find more
information about this [here](nabu/computing/README.md)
- computing [default: standart]: the distributed computing software you want to
use. One of standart or condor. standart means that no distributed computing
software is used and the job will run on the machine where nabu is called from.
the condor option uses HTCondor. More information can be found
[here](nabu/computing/README.md).

### Testing

In the testing stage the performance of the model is evaluated on a testing set.
To modify the way the model in is evaluated you can modify the
test_evaluator.cfg file in the recipe dir. You can find more information on
evaluators [here](nabu/neuralnetworks/trainers/README.md).

You can run testing with

```
run test --recipe=/path/to/recipe --expdir=/path/to/expdir
--computing=<computing>
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
run decode --recipe=/path/to/recipe --expdir=/path/to/expdir
--computing=<computing>
```

The parameters for this script are similar to the training script (see above).
You should use the same expdir that you used for training the model.

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
- Add your file to the package in \__init\__.py
- create a configuration file for your class and put it in templates. You
should then add this configuration file in whichever recipe you want to use it
for or create your own recipe using your new class.

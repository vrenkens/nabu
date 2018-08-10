# Recipes

A recipe contains configuration files for training and testing models designed
for a specific database. Nabu alleady contains several pre-designed recipes,
but you can design your own recipes with little effort. A recipe contains
the following configurations files:

- database.conf: This is the database configuration. For every set of data
(training features, training text, testing features ...) it contains a section
that specifies where to read and write the data in your file system and wich
processors should be used to process the data. Every recipe contains a template
for the database configuration, but because the pahs are different ofor each
person this has to be completed. You can put the completed database configuation
in database.conf (which will automatically be added to the gitignore)
- processor configuration files: One or multiple files for the data processors
that are used. The database confiuration will point to these processor
confiurations. For example, a feature processor specifies the type
of feature, the feature parameters (window length, numbe of filters, ...) etc.
You can find more information about processors
[here](../../nabu/processing/processors/README.md).
- model.cfg: This is the model configuration. It specifies
the model parameters (number of layers, units ...). The classifies
configuration contains 3 sections:
  - io: give names to the inputs and outputs of the model
  - encoder: the encoder configuration. The encoder encodes the input to
  a new representation.
  - decoder: the decoder configuration. The decoder decodes the encoder
  representation into the desired output.
You can find more information about models
[here](../../nabu/neuralnetworks/models/README.md).
- trainer.cfg: specifies the trainer parameters (learning rate, nuber of epochs,
...). This configuration also specifies which data should be used to train
the model. It links io names defined in the model configuration to
data sections defined in the database configuration. A trainer is used to
train the model to minimize some loss function. You can find more
information about trainers
[here](../../nabu/neuralnetworks/trainers/README.md).
- validation_evaluator.cfg: specifies the validator type and parameters,
that will be used during training. It also specifies which data will be used
to evaluate the model in the same way the trainer configuration does.
An evaluator is used to evaluate the performance of a model. The
validation evaluator is used during training to measure the performance at fixed
intervals and adjusts the learning rate if necesarry. You can find more
information about evaluators
[here](../../nabu/neuralnetworks/evaluators/README.md)
- test_evaluator.cfg: This is the configuartion for the evaluator to be used at
test time (see validation_evaluator.cfg)
- recognizer.cfg: The configuration for the recognizer to be used for decoding
it contains similar fields as the evaluator configs.

To create your own recipe, simply create a directory containing all of the
mentioned configuation files. You can find template configuations in
config/templates.

All the components in the recipe have default configurations. You can find these
configurations in the directory where the component is implemented under the
defaults directory. If a field is not defined in the configuration it will be
filled in with the default value.

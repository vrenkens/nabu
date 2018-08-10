# Evaluators

An evaluator is used to evaluate the performance of the model during training
or at test time. To create a new evaluator you should inherit from the general
Evaluator class defined in evaluator.py and overwrite all the abstract methods.
Afterwards you should add it to the factory method in evaluator_factory.py.
It is also very helpful to create a default configuration in the defaults 
directory. The name of the file should be the name of the class in lower case
with the .cfg extension.

The decoder_evaluator will use a decoder to decode the validation set and
compare the results with the ground truth. You can find more information about
decoders [here](../decoders/README.md).

# Trainers

A trainer is used to update the model parameters to minimize some loss function.
To create a new trainer you should inherit from the general Trainer class
defined in trainer.py and overwrite the abstract methods. Afterwards yo should
add the trainer to the factory method in trainer_factory.py.
It is also very helpful to create a default configuration in the defaults
directory. The name of the file should be the name of the class in lower case
with the .cfg extension.

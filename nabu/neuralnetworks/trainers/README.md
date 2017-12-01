# Trainers

A trainer is used to update the model parameters to minimize some loss function.
To create a new trainer you should inherit from the general Trainer class
defined in trainer.py and overwrite the abstract methods. Afterwards yo should
add the trainer to the factory method in trainer_factory.py and the package in
\_\_init\_\_.py

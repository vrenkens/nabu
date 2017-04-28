# Target Normalizers

A target normalizer is used to normalize targets to make them usable for
training. Some examples of normalization steps are replacing unknown
characters with a fixed label or making everyting lower case. A target
normalizer is different for each database. To create a new target normalizer
you should inherit from the general Normalizer class defined in normalizer.py
and overwrite the abstract methods. You should then add it to the factory method
in normalizer_factory.py and to the package in __init__.py.

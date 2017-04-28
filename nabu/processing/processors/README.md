# Processors

A processor is used to process data in the data preperation script. For example,
feature computation for audio or normalization for text. To create a new
processor you should inherit from the general Processor class defined in
processor.py and overwrite the abstract methods. You should then add it to the
factory method in processor_factory.py and to the package in __init__.py.

You can find more information about feature computers
[here](../feature_computers/README.md) and target normalizers
[here](../target_normalizers/README.md)

# Processors

A processor is used to process data in the data preperation script. For example,
feature computation for audio or normalization for text. To create a new
processor you should inherit from the general Processor class defined in
processor.py and overwrite the abstract methods. You should then add it to the
factory method in processor_factory.py.
It is also very helpful to create a default configuration in the defaults
directory. The name of the file should be the name of the class in lower case
with the .cfg extension.

You can find more information about feature computers
[here](../feature_computers/README.md) and target normalizers
[here](../target_normalizers/README.md)

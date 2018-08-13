# TF Writers

A TensorFlow writer is used to write processed data to TFRecord files in the
data preparation. To create a new writer you should inherit from the general
TFWriter class defined in tfwriter.py and overwrite the abstract methods. You
should then add it to the factory method in tfwriter_factory.py.
It is also very helpful to create a default configuration in the defaults
directory. The name of the file should be the name of the class in lower case
with the .cfg extension.

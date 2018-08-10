# TF Readers

a TensorFlow Reader is used to read TFRecord files directly into the graph.
To create a new TFReader you should inherit from the general TFReader class
defined in tfreader.py and overwrite the abstract methods. You should then add
it to the factory method in tfreader_factory.py.
It is also very helpful to create a default configuration in the defaults
directory. The name of the file should be the name of the class in lower case
with the .cfg extension.

# TF Readers

a TensorFlow Reader is used to read TFRecord files directly into the graph.
To create a new TFReader you should inherit from the general TFReader class
defined in tfreader.py and overwrite the abstract methods. You should then add
it to the factory method in tfreader_factory.py and to the package in
\_\_init\_\_.py.

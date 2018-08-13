# Decoders

A Decoder uses a model to convert the inputs to sequences of output labels.
To create a new decoder you should inherit from the general Decoder class
defined in decoder.py and overwrite the abstract methods. Afterwards you should
add the decoder to the factory method in decoder_factory.py. It is also very
helpful to create a default configuration in the defaults directory. The name of
the file should be the name of the class in lower case with the .cfg extension.

# Decoders

A Decoder uses a model to convert the inputs to sequences of output labels.
To create a new decoder you should inherit from the general Decoder class
defined in decoder.py and overwrite the abstract methods. Afterwards you should
add the decoder to the factory method in decoder_factory.py and the package in
\__init\__.py.

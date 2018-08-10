# Encoder-Decoder Decoder

An encoder-decoder decoder (ed-decoder) outputs labels based on the hidden
representation of the encoder and the history of decoded labels. To create a new
ed-decoder you should inherit from the general EDDecoder class defined in
ed-decoder.py and overwrite the abstract methods.
Afterwards you should add it to the factory method in ed_decoder_factory.py.
It is also very helpful to create a default configuration in the defaults
directory. The name of the file should be the name of the class in lower case
with the .cfg extension.

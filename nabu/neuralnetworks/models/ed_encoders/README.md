# Encoder-Decoder Encoders

an encoder-decoder encoder (ed-encoder) encoder the inputs into a hidden
representation. To create a new ed-encoder you should inherit from the general
EDEncoder class defined in ed_encoder.py and overwrite the abstract methods.
Afterwards you should add it to the factory method in ed_encoder_factory.py.
It is also very helpful to create a default configuration in the defaults
directory. The name of the file should be the name of the class in lower case
with the .cfg extension.

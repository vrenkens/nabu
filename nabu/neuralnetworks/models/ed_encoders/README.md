# Encoder-Decoder Encoders

an encoder-decoder encoder (ed-encoder) encoder the inputs into a hidden
representation. To create a new ed-encoder you should inherit from the general
EDEncoder class defined in ed_encoder.py and overwrite the abstract methods.
Afterwards you should add it to the factory method in ed_encoder_factory.py and
to the package in \_\_init\_\_.py.

# Target Normalizers

A target normalizer is used to normalize targets to make them usable for
training. Some examples of normalization steps are replacing unknown
characters with a fixed label or making everyting lower case. A target
normalizer can be different for each database. To create a new target normalizer
you should create a file with normalize method that takes a transcription and
an alphabet as input and returns the normalized transcription. You should then
add it to the factory method in normalizer_factory.py and to the package in
\_\_init\_\_.py.

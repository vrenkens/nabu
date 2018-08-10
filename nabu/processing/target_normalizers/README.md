# Target Normalizers

A target normalizer is used to normalize targets to make them usable for
training. Some examples of normalization steps are replacing unknown
characters with a fixed label or making everyting lower case. A target
normalizer can be different for each database. To create a new target normalizer
you should create a file with normalize method that takes a transcription and
an alphabet as input and returns the normalized transcription. You should then
add it to the factory method in normalizer_factory.py.
It is also very helpful to create a default configuration in the defaults
directory. The name of the file should be the name of the class in lower case
with the .cfg extension.

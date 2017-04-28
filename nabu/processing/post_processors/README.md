# Post Processors

The post processor does some additional processing on the output of the network
before writing the result to disk. This processing happens ouside the graph.
It can be used for example to convert the sequences of output labels back to
text. to create a new post_processor you should inherit from the general
PostProcessor class defined in post_processor.py and overwrite the abstract
methods. You should then add it to the factory methor in
post_processor_factory.py and to the package in \_\_init\_\_.py

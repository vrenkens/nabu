'''@file recognizer.py
contains the Recognizer class'''

import os
import shutil
import math
import tensorflow as tf
from nabu.processing import input_pipeline
from nabu.neuralnetworks.models.model import Model
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.processing.post_processors import post_processor_factory
from nabu.neuralnetworks.components.hooks import LoadAtBegin, SummaryHook

class Recognizer(object):
    '''a Recognizer can use a model to produce nbest lists for an input'''

    def __init__(self, conf, modelconf, dataconf, expdir):
        '''Recognizer constructor

        Args:
            conf: the recognizer configuration as a configparser
            modelconf: the model configuration as a configparser
            dataconf: the database configuration as a configparser
            expdir: the experiments directory
        '''

        self.conf = conf
        decoderconf = dict(conf.items('decoder'))
        postprocessorconf = dict(conf.items('post_processor'))
        self.expdir = expdir

        #get the database configurations
        inputs = modelconf.get('io', 'inputs').split(' ')
        if inputs == ['']:
            inputs = []
        input_sections = [conf.get('recognizer', i) for i in inputs]
        self.input_dataconfs = []
        for section in input_sections:
            self.input_dataconfs.append(dict(dataconf.items(section)))

        #create the post_processor
        self.post_processor = post_processor_factory.factory(
            postprocessorconf['post_processor'])(postprocessorconf)

        #get the decoder class
        decoder_class = decoder_factory.factory(decoderconf['decoder'])
        output_dims = decoder_class.get_output_dims(
            [self.post_processor.num_labels])

        #create the model
        self.model = Model(
            conf=modelconf,
            output_dims=output_dims)

        #create a decoder
        self.decoder = decoder_class(decoderconf, self.model)

        batch_size = int(self.conf.get('recognizer', 'batch_size'))

        #create the graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            #get the list of filenames fo the validation set
            data_queue_elements, self.names = input_pipeline.get_filenames(
                self.input_dataconfs)

            #compute the number of batches in the validation set
            self.numbatches = int(math.ceil(
                float(len(data_queue_elements))/batch_size))

            #create a queue to hold the filenames
            data_queue = tf.train.string_input_producer(
                string_tensor=data_queue_elements,
                num_epochs=1,
                shuffle=False,
                seed=None,
                capacity=batch_size*2)

            #create the input pipeline
            inputs, input_seq_length = input_pipeline.input_pipeline(
                data_queue=data_queue,
                batch_size=batch_size,
                numbuckets=1,
                allow_smaller_final_batch=True,
                dataconfs=self.input_dataconfs
            )

            self.decoded = self.decoder(inputs, input_seq_length)

    def recognize(self):
        '''perform the recognition'''

        with self.graph.as_default():
            #create a hook that will load the model
            load_hook = LoadAtBegin(os.path.join(
                self.expdir, 'model', 'network.ckpt'))

            #create a hook for summary writing
            summary_hook = SummaryHook(os.path.join(self.expdir, 'logdir'))

            if os.path.isdir(os.path.join(self.expdir, 'decoded')):
                shutil.rmtree(os.path.join(self.expdir, 'decoded'))
            os.makedirs(os.path.join(self.expdir, 'decoded'))

            #start the session
            with tf.train.SingularMonitoredSession(
                hooks=[load_hook, summary_hook]) as sess:

                nameid = 0
                for _ in range(self.numbatches):
                    #decode
                    outputs, scores = sess.run(self.decoded)

                    #postprocessing
                    processed = self.post_processor(outputs, scores)

                    #write to disk
                    names = self.names[nameid:nameid+len(processed)]
                    nameid += len(processed)
                    for i, p in enumerate(processed):
                        self.post_processor.write(
                            p, os.path.join(self.expdir, 'decoded'), names[i])

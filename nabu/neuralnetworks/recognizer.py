'''@file recognizer.py
contains the Recognizer class'''

import os
import shutil
import math
import cPickle as pickle
import tensorflow as tf
from nabu.processing import input_pipeline
from nabu.neuralnetworks.decoders import decoder_factory
from nabu.neuralnetworks.components.hooks import LoadAtBegin, SummaryHook

class Recognizer(object):
    '''a Recognizer can use a model to produce decode

    stores the results on disk'''

    def __init__(self, conf, dataconf, expdir):
        '''Recognizer constructor

        Args:
            conf: the recognizer configuration as a configparser
            modelconf: the model configuration as a configparser
            dataconf: the database configuration as a configparser
            expdir: the experiments directory
        '''

        self.conf = conf
        self.expdir = expdir

        #load the model
        with open(os.path.join(expdir, 'model', 'model.pkl'), 'rb') as fid:
            self.model = pickle.load(fid)

        #get the database configurations
        input_sections = [conf.get('recognizer', i)
                          for i in self.model.input_names]
        self.input_dataconfs = []
        for section in input_sections:
            self.input_dataconfs.append(dict(dataconf.items(section)))

        decoderconf = dict(conf.items('decoder'))

        #create a decoder
        self.decoder = decoder_factory.factory(decoderconf['decoder'])(
            decoderconf, self.model)

        self.batch_size = int(self.conf.get('recognizer', 'batch_size'))

        #create the graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            #get the list of filenames fo the validation set
            data_queue_elements, self.names = input_pipeline.get_filenames(
                self.input_dataconfs)

            #compute the number of batches in the validation set
            self.numbatches = int(math.ceil(
                float(len(data_queue_elements))/self.batch_size))

            #create a queue to hold the filenames
            data_queue = tf.train.string_input_producer(
                string_tensor=data_queue_elements,
                num_epochs=1,
                shuffle=False,
                seed=None,
                capacity=self.batch_size*2)

            #create the input pipeline
            inputs, input_seq_length = input_pipeline.input_pipeline(
                data_queue=data_queue,
                batch_size=self.batch_size,
                numbuckets=1,
                allow_smaller_final_batch=True,
                dataconfs=self.input_dataconfs
            )

            inputs = {
                self.model.input_names[i]: d
                for i, d in enumerate(inputs)}
            input_seq_length = {
                self.model.input_names[i]: d
                for i, d in enumerate(input_seq_length)}

            self.decoded = self.decoder(inputs, input_seq_length)

    def recognize(self):
        '''perform the recognition'''

        with self.graph.as_default():
            #create a hook that will load the model
            load_hook = LoadAtBegin(
                os.path.join(self.expdir, 'model', 'network.ckpt'),
                self.model)

            #create a hook for summary writing
            summary_hook = SummaryHook(os.path.join(self.expdir, 'logdir'))

            directory = os.path.join(self.expdir, 'decoded')
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

            #start the session
            with tf.train.SingularMonitoredSession(
                hooks=[load_hook, summary_hook]) as sess:

                nameid = 0
                for _ in range(self.numbatches):
                    #decode
                    outputs = sess.run(self.decoded)

                    #write to disk
                    names = self.names[nameid:nameid+self.batch_size]
                    self.decoder.write(outputs, directory, names)
                    nameid += self.batch_size

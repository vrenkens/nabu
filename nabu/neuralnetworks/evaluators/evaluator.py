'''@file evaluator.py
contains the Evaluator class'''

import os
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from nabu.processing import input_pipeline
from nabu.neuralnetworks.models.model import Model

class Evaluator(object):
    '''the general evaluator class

    an evaluator is used to evaluate the performance of a model'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, dataconf, model_or_conf):
        '''Evaluator constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            model_or_conf: eihther a model object or a model
                configuration as a configparser
        '''

        self.conf = conf

        if isinstance(model_or_conf, Model):
            self.model = model_or_conf

            #get the database configurations
            inputs = self.model.input_names
            input_sections = [conf.get('evaluator', i) for i in inputs]
            self.input_dataconfs = []
            for section in input_sections:
                self.input_dataconfs.append(dict(dataconf.items(section)))
            outputs = self.model.output_names
            target_sections = [conf.get('evaluator', o) for o in outputs]
            self.target_dataconfs = []
            for section in target_sections:
                self.target_dataconfs.append(dict(dataconf.items(section)))

        else:
            #get the database configurations
            inputs = model_or_conf.get('io', 'inputs').split(' ')
            if inputs == ['']:
                inputs = []
            input_sections = [conf.get('evaluator', i) for i in inputs]
            self.input_dataconfs = []
            for section in input_sections:
                self.input_dataconfs.append(dict(dataconf.items(section)))
            outputs = model_or_conf.get('io', 'outputs').split(' ')
            if outputs == ['']:
                outputs = []
            target_sections = [conf.get('evaluator', o) for o in outputs]
            self.target_dataconfs = []
            for section in target_sections:
                self.target_dataconfs.append(dict(dataconf.items(section)))

            #get the dimensions of all the targets
            output_dims = []
            for c in self.target_dataconfs:
                with open(os.path.join(c['dir'], 'dim')) as fid:
                    output_dims.append(int(fid.read()))

            #adjust the output dimensions if necesary
            output_dims = self.get_output_dims(output_dims)

            #create the model
            self.model = Model(
                conf=model_or_conf,
                output_dims=output_dims)

    def evaluate(self):
        '''evaluate the performance of the model

        Returns:
            - the loss as a scalar tensor
            - the number of batches in the validation set as an integer
        '''

        batch_size = int(self.conf.get('evaluator', 'batch_size'))

        with tf.name_scope('evaluate'):

            #get the list of filenames fo the validation set
            data_queue_elements, _ = input_pipeline.get_filenames(
                self.input_dataconfs + self.target_dataconfs)

            #compute the number of batches in the validation set
            numbatches = len(data_queue_elements)/batch_size

            #create a queue to hold the filenames
            data_queue = tf.train.string_input_producer(
                string_tensor=data_queue_elements,
                shuffle=False,
                seed=None,
                capacity=batch_size*2)

            #create the input pipeline
            data, seq_length = input_pipeline.input_pipeline(
                data_queue=data_queue,
                batch_size=batch_size,
                numbuckets=1,
                dataconfs=self.input_dataconfs + self.target_dataconfs
            )

            inputs = data[:len(self.input_dataconfs)]
            input_seq_length = seq_length[:len(self.input_dataconfs)]
            targets = data[len(self.input_dataconfs):]
            target_seq_length = seq_length[len(self.input_dataconfs):]

            loss = self.compute_loss(inputs, input_seq_length, targets,
                                     target_seq_length)

        return loss, numbatches

    @abstractmethod
    def compute_loss(self, inputs, input_seq_length, targets,
                     target_seq_length):
        '''compute the validation loss for a batch of data

        Args:
            inputs: the inputs to the neural network, this is a list of
                [batch_size x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a list of [batch_size] vectors
            targets: the targets to the neural network, this is a list of
                [batch_size x max_output_length] tensors.
            target_seq_length: The sequence lengths of the target utterances,
                this is a list of [batch_size] vectors

        Returns:
            the loss as a scalar'''

    @abstractmethod
    def get_output_dims(self, output_dims):
        '''
        Adjust the output dimensions of the model (blank label, eos...)

        Args:
            a list containing the original model output dimensions

        Returns:
            a list containing the new model output dimensions
        '''

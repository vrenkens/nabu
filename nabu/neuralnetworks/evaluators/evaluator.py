'''@file evaluator.py
contains the Evaluator class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from nabu.processing import input_pipeline

class Evaluator(object):
    '''the general evaluator class

    an evaluator is used to evaluate the performance of a model'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, dataconf, model):
        '''Evaluator constructor

        Args:
            conf: the evaluator configuration as a ConfigParser
            dataconf: the database configuration
            model: the model to be evaluated
        '''

        self.conf = conf
        self.model = model

        #get the database configurations
        inputs = self.model.input_names
        input_sections = [conf.get('evaluator', i) for i in inputs]
        self.input_dataconfs = []
        for section in input_sections:
            self.input_dataconfs.append(dict(dataconf.items(section)))

        target_sections = conf.get('evaluator', 'targets').split(' ')
        self.target_dataconfs = []
        for section in target_sections:
            self.target_dataconfs.append(dict(dataconf.items(section)))

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

            inputs = {
                self.model.input_names[i]: d
                for i, d in enumerate(data[:len(self.input_dataconfs)])}

            input_seq_length = {
                self.model.input_names[i]: d
                for i, d in enumerate(seq_length[:len(self.input_dataconfs)])}

            targets = {
                self.model.output_names[i]: d
                for i, d in enumerate(data[len(self.input_dataconfs):])}

            target_seq_length = {
                self.model.output_names[i]: d
                for i, d in enumerate(seq_length[len(self.input_dataconfs):])}

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

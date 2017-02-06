'''@file decoder.py
neural network decoder environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np

class Decoder(object):
    '''the abstract class for a decoder'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, classifier, classifier_scope, input_dim,
                 max_input_length, coder, expdir):
        '''
        Decoder constructor, creates the decoding graph

        Args:
            conf: the decoder config
            classifier: the classifier that will be used for decoding
            classifier_scope: the scope where the classier should be created/loaded
                from
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximum length of the inputs
            batch_size: the decoder batch size
            coder: a TargetCoder object
            expdir: the location where the models were saved and the results
                will be written
        '''

        self.conf = conf
        self.max_input_length = max_input_length
        self.expdir = expdir
        self.coder = coder
        self.batch_size = int(conf['batch_size'])
        with tf.variable_scope(type(self).__name__):
            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32,
                shape=[self.batch_size, max_input_length, input_dim],
                name='inputs')

            #create the sequence length placeholder
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[self.batch_size], name='seq_length')

            #compute the outputs
            self.outputs = self.get_outputs(
                inputs=self.inputs,
                input_seq_length=self.input_seq_length,
                classifier=classifier,
                classifier_scope=classifier_scope
                )

    @abstractmethod
    def get_outputs(self, inputs, input_seq_length, classifier,
                    classifier_scope):

        '''compute the outputs of the decoder

        Args:
            inputs: The inputs to the network as a
                [batch_size x max_input_length x input_dim] tensor
            input_seq_length: The sequence length of the inputs as a
                [batch_size] vector
            classifier: The classifier object that will be used in decoding
            classifier_scope: the scope where the classifier was defined

        Returns:
            A list with batch_size elements containing nbest lists with elements
            containing pairs of score and output labels
        '''

    @abstractmethod
    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing nbest lists of decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''


    def decode(self, reader, sess):
        '''decode using the neural net

        Args:
            decoder: the decoder that should be used
            reader: a feauture reader object containing the testng features
            sess: a tensorflow session

        Returns:
            a dictionary containing the outputs
        '''

        #start the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        config.allow_soft_placement = True

        decoded = dict()
        looped = False

        while not looped:

            utt_ids = []
            inputs = []

            for _ in range(self.batch_size):
                #read a batch of data
                (utt_id, inp, looped) = reader.get_utt()

                if looped:
                    reader.prev_id()
                    break

                utt_ids.append(utt_id)
                inputs.append(inp)

            if len(utt_ids) == 0:
                break

            #add empty elements to the inputs to get a full batch
            feat_dim = inputs[0].shape[1]
            inputs += [np.zeros([0, feat_dim])]*(self.batch_size-len(inputs))

            #get the sequence length
            input_seq_length = [inp.shape[0] for inp in inputs]

            #pad the inputs and put them in a tensor
            inputs = np.array([np.append(
                inp, np.zeros([self.max_input_length-inp.shape[0],
                               inp.shape[1]]), 0) for inp in inputs])

            #pylint: disable=E1101
            output = sess.run(
                self.outputs,
                feed_dict={self.inputs:inputs,
                           self.input_seq_length:input_seq_length})

            #convert the label sequence into a sequence of characers
            for i, utt_id in enumerate(utt_ids):
                decoded[utt_id] = [(p[0], self.coder.decode(p[1]))
                                   for p in output[i]]

        return decoded

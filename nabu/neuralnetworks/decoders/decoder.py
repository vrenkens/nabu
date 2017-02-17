'''@file decoder.py
neural network decoder environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np

class Decoder(object):
    '''the abstract class for a decoder'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, classifier, input_dim,
                 max_input_length, coder, expdir):
        '''
        Decoder constructor, creates the decoding graph

        Args:
            conf: the decoder config
            classifier: the classifier that will be used for decoding
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
                classifier=classifier)

    @abstractmethod
    def get_outputs(self, inputs, input_seq_length, classifier):

        '''compute the outputs of the decoder

        Args:
            inputs: The inputs to the network as a
                [batch_size x max_input_length x input_dim] tensor
            input_seq_length: The sequence length of the inputs as a
                [batch_size] vector
            classifier: The classifier object that will be used in decoding

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
            reader: a feauture reader object containing the testng features
            sess: a tensorflow session

        Returns:
            a dictionary containing the outputs
        '''

        decoded = dict()
        looped = False

        while not looped:

            utt_ids = []
            inputs = []

            for _ in range(self.batch_size):
                #read a batch of data
                (utt_id, inp, looped) = reader.get_utt()

                utt_ids.append(utt_id)
                inputs.append(inp)

                if looped:
                    break

            #add empty elements to the inputs to get a full batch
            feat_dim = inputs[0].shape[1]
            inputs += [np.zeros([0, feat_dim])]*(self.batch_size-len(inputs))

            #get the sequence length
            input_seq_length = [inp.shape[0] for inp in inputs]

            #pad the inputs and put them in a tensor
            input_tensor = np.array([np.append(
                inp, np.zeros([self.max_input_length-inp.shape[0],
                               inp.shape[1]]), 0) for inp in inputs])

            #pylint: disable=E1101
            output = sess.run(
                self.outputs,
                feed_dict={self.inputs:input_tensor,
                           self.input_seq_length:input_seq_length})

            #convert the label sequence into a sequence of characers
            for i, utt_id in enumerate(utt_ids):
                decoded[utt_id] = [(p[0], self.coder.decode(p[1]))
                                   for p in output[i]]

        return decoded

    def decode_utt(self, features, sess):
        '''decode a list of utterances

        Args:
            features: the input features as a list of [seq_length, feature_dim]
                numpy arrays
            sess: a tensorflow session

        Returns:
            the decoded utterance as a string
        '''

        #get the sequecnce length
        input_seq_length = features.shape[0]

        #pad the features
        inputs = np.append(
            features,
            np.zeros([self.max_input_length-features.shape[0],
                      features.shape[1]]),
            0)

        #put the inputs in the correct shape
        inputs = inputs[np.newaxis, :, :]
        input_seq_length = np.array([input_seq_length])

        #decode the utterance
        output = sess.run(
            self.outputs,
            feed_dict={self.inputs:inputs,
                       self.input_seq_length:input_seq_length})

        #get the scores of the beam elements
        scores = np.array([h[0] for h in output[0]])

        #get the best label sequences in the beam
        decoded = output[0][np.argmax(scores)][1]

        #convert the label sequence to a target sequence
        decoded = self.coder.decode(decoded)

        return decoded

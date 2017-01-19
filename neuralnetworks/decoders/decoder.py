'''@file decoder.py
neural network decoder environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np

class Decoder(object):
    '''the abstract class for a decoder'''

    __metaclass__ = ABCMeta

    @abstractmethod
    def process_decoded(self, decoded):
        '''
        do some postprocessing on the output of the decoding graph

        Args:
            decoded: the outputs of the decoding graph

        Returns:
            a list of pairs containing:
                - the score of the output
                - the output lable sequence as a numpy array
        '''

    @abstractmethod
    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''


def decode(decoder, reader, sess):
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

    while True:

        (utt_id, inputs, looped) = reader.get_utt()

        if looped:
            reader.prev_id()
            break

        #get the sequence length
        input_seq_length = [inputs.shape[0]]

        #pad the inputs
        inputs = np.append(
            inputs, np.zeros([decoder.max_input_length-inputs.shape[0],
                              inputs.shape[1]]), 0)

        #pylint: disable=E1101
        output = sess.run(
            decoder.outputs,
            feed_dict={decoder.inputs:inputs[np.newaxis, :, :],
                       decoder.input_seq_length:input_seq_length})

        processed = decoder.process_decoded(output)

        #convert the label sequence into a sequence of characers
        decoded[utt_id] = [(p[0], decoder.coder.decode(p[1]))
                           for p in processed]

    return decoded

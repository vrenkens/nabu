'''@file decoder.py
neural network decoder environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np

class Decoder(object):
    '''the abstract class for a decoder'''

    __metaclass__ = ABCMeta

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
    looped = False

    while not looped:

        utt_ids = []
        inputs = []

        for _ in range(decoder.batch_size):
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
        inputs += [np.zeros([0, feat_dim])]*(decoder.batch_size-len(inputs))

        #get the sequence length
        input_seq_length = [inp.shape[0] for inp in inputs]

        #pad the inputs and put them in a tensor
        inputs = np.array([np.append(
            inp, np.zeros([decoder.max_input_length-inp.shape[0],
                           inp.shape[1]]), 0) for inp in inputs])

        #pylint: disable=E1101
        output = sess.run(
            decoder.outputs,
            feed_dict={decoder.inputs:inputs,
                       decoder.input_seq_length:input_seq_length})

        #convert the label sequence into a sequence of characers
        for i, utt_id in enumerate(utt_ids):
            decoded[utt_id] = [(p[0], decoder.coder.decode(p[1]))
                               for p in output[i]]

    return decoded

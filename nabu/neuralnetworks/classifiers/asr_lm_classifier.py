'''@file asr_lm_classifier.py
contains the AsrLmClassifier class'''

from collections import namedtuple
import tensorflow as tf
import classifier
from asr import asr_factory
from lm import lm_factory

class AsrLmClassifier(classifier.Classifier):
    '''a classifier that combines an endoder-decoder asr with
    a language model'''

    def __init__(self, conf, asr_conf, lm_conf, output_dim, name=None):
        '''classifier constructor

        Args:
            conf: The classifier configuration
            asr_conf: the asr configuration
            lm_conf: the lm configuration
            output_dim: the classifier output dimension
            name: the classifier name
        '''

        #create the language model
        lm = lm_factory.factory(lm_conf, output_dim)

        #create the asr
        asr = asr_factory.factory(asr_conf, output_dim)

        #create the encoder
        self.encoder = asr.encoder

        #create the decoder
        self.decoder = AsrLmDecoder(asr.decoder, lm.decoder,
                                    float(conf['lm_weight']))

        super(AsrLmClassifier, self).__init__(conf, output_dim)

    def _get_outputs(self, inputs, input_seq_length, targets=None,
                     target_seq_length=None, is_training=False):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        #add input noise
        std_input_noise = float(self.conf['std_input_noise'])
        if is_training and std_input_noise > 0:
            noisy_inputs = inputs + tf.random_normal(
                inputs.get_shape(), stddev=std_input_noise)
        else:
            noisy_inputs = inputs

        #compute the high level features
        hlfeat = self.encoder(
            inputs=noisy_inputs,
            sequence_lengths=input_seq_length,
            is_training=is_training)

        #prepend a sequence border label to the targets to get the encoder
        #inputs, the label is the last label
        batch_size = int(targets.get_shape()[0])
        s_labels = tf.constant(self.output_dim-1,
                               dtype=tf.int32,
                               shape=[batch_size, 1])
        encoder_inputs = tf.concat(1, [s_labels, targets])

        #compute the output logits
        logits, _ = self.decoder(
            hlfeat=hlfeat,
            encoder_inputs=encoder_inputs,
            numlabels=self.output_dim,
            initial_state=None,
            is_training=is_training)

        return logits, target_seq_length + 1

class AsrLmDecoder(object):
    '''a decoder that combines an asr decoder with a lm decoder'''

    def __init__(self, asr_decoder, lm_decoder, lm_weight):
        '''
        AsrLmDecoder constructor

        Args:
            asr_decoder: the asr decoder object
            lm_decoder: the language model decoder object
            lm_weight: a value between 0 and 1, the weight of the language model
        '''

        self.asr_decoder = asr_decoder
        self.lm_decoder = lm_decoder
        self.lm_weight = lm_weight

    def __call__(self, hlfeat, encoder_inputs, numlabels, initial_state=None,
                 initial_state_attention=False, is_training=False):
        '''
        Create the variables and do the forward computation

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim]
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length].
            numlabels: number of output labels
            initial_state: the initial decoder state, could be usefull for
                decoding
            initial_state_attention: whether attention has to be applied
                to the initital state to ge an initial context
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the decoder as a
                [batch_size x target_seq_length x numlabels] tensor
            - the final state of the listener
        '''

        if initial_state is None:
            initial_state = self.zero_state(hlfeat.get_shape[0])

        #compute the logits for the asr decoder
        asr_logits, asr_state = self.asr_decoder(
            hlfeat=hlfeat,
            encoder_inputs=encoder_inputs,
            numlabels=numlabels,
            initial_state=initial_state.asr,
            initial_state_attention=initial_state_attention,
            is_training=is_training)

        #compute the lm logits
        lm_logits, lm_state = self.lm_decoder(
            encoder_inputs=encoder_inputs,
            numlabels=numlabels,
            initial_state=initial_state.lm,
            is_training=is_training)

        #combine the two logits
        logits = (1-self.lm_weight)*asr_logits + self.lm_weight*lm_logits

        state = AsrLmState(asr_state, lm_state)

        return logits, state

    def zero_state(self, batch_size):
        '''get the listener zero state

        Args:
            batch_size: the batch size

        Returns:
            an rnn_cell zero state'''

        asr_state = self.asr_decoder.zero_state(batch_size)
        lm_state = self.lm_decoder.zero_state(batch_size)

        return AsrLmState(asr_state, lm_state)

class AsrLmState(namedtuple('AsrLmState', ['asr', 'lm'])):
    '''a named tupple class for a joint asr lm state'''
    pass

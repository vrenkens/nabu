'''@ file phonology_decoder.py
contains the PhonologyDecoder class'''

import numpy as np
import tensorflow as tf
import ed_decoder
import ed_decoder_factory

class PhonologyDecoder(ed_decoder.EDDecoder):
    '''a decoder that is used to detect phonological features'''

    def __init__(self, conf, trainlabels, outputs, name=None):
        '''EDDecoder constructor

        Args:
            conf: the decoder configuration
            trainlabels: the number of extra labels required by the trainer
            outputs: the name of the outputs of the model
        '''

        #create the wrapped decoder
        self.wrapped = ed_decoder_factory.factory(conf['wrapped'])(
            conf, trainlabels, conf['output_names'].split(' '))

        #read the mapping
        with open(conf['mapping'], 'rb') as fid:
            #load the mapping
            mapping = np.load(fid)

        #add the blank mapping
        extra_mappings = []
        feature_names = conf['output_names'].split(' ')
        for o in feature_names:
            numfeats = self.wrapped.output_dims[o]
            extra_mapping = np.zeros([numfeats, trainlabels])
            for i in range(trainlabels):
                extra_mapping[-i-1, i] = 1
            extra_mappings.append(extra_mapping)
        extra_mapping = np.concatenate(extra_mappings, 0)
        self.mapping = np.concatenate([mapping, extra_mapping], 1)
        self.mapping = self.mapping.astype(np.float32)

        #super constructor
        super(PhonologyDecoder, self).__init__(conf, trainlabels, outputs, name)

    def _decode(self, encoded, encoded_seq_length, targets, target_seq_length,
                is_training):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a dictionary of
                [batch_size x time x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a dictionary of [batch_size] vectors
            targets: the targets used as decoder inputs as a dictionary of
                [batch_size x time x ...] tensors
            target_seq_length: the sequence lengths of the targets
                as a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the decoder as a dictionary of
                [batch_size x time x ...] tensors
            - the logit sequence_lengths as a dictionary of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

        #call the wrapped decoder
        logits, logit_seq_length, state = self.wrapped(
            encoded, encoded_seq_length, targets, target_seq_length,
            is_training)

        #concatenate the outputs
        feature_names = self.conf['output_names'].split(' ')
        feat_logits = tf.concat([logits[f] for f in feature_names], axis=2)

        #map the feature logits to phone logits
        outputs = {self.outputs[0]:tf.tensordot(feat_logits, self.mapping, 1)}
        output_seq_length = {self.outputs[0]:logit_seq_length.values()[0]}

        return outputs, output_seq_length, state


    def _step(self, encoded, encoded_seq_length, targets, state, is_training):
        '''take a single decoding step

        encoded: the encoded inputs, this is a dictionary of
            [batch_size x time x ...] tensors
        encoded_seq_length: the sequence lengths of the encoded inputs
            as a dictionary of [batch_size] vectors
        targets: the targets decoded in the previous step as a dictionary of
            [batch_size] vectors
        state: the state of the previous deocding step as a possibly nested
            tupple of [batch_size x ...] vectors
        is_training: whether or not the network is in training mode.

        Returns:
            - the output logits of this decoding step as a dictionary of
                [batch_size x ...] tensors
            - the updated state as a possibly nested tupple of
                [batch_size x ...] vectors
        '''

        #call the wrapped decoder
        logits, state = self.wrapped.step(
            encoded, encoded_seq_length, targets, state, is_training)

        #concatenate the outputs
        feature_names = self.conf['output_names'].split(' ')
        feat_logits = tf.concat([logits[f] for f in feature_names], axis=1)

        #map the feature logits to phone logits
        outputs = {self.outputs[0]:tf.matmul(feat_logits, self.mapping)}

        return outputs, state

    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded sequence as a dictionary
                of integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors'''

        return self.wrapped.zero_state(encoded_dim, batch_size)

    def get_output_dims(self, trainlabels):
        '''get the decoder output dimensions

        args:
            targetconfs: the target data confs
            trainlabels: the number of extra labels the trainer needs

        returns:
            a dictionary containing the output dimensions'''

        name = self.outputs[0]
        dim = self.mapping.shape[1]

        #get the dimensions of all the targets
        output_dims = {name:dim}

        return output_dims

    @property
    def variables(self):
        '''get a list of the models's variables'''

        ownvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=self.scope.name)

        return ownvars + self.wrapped.variables

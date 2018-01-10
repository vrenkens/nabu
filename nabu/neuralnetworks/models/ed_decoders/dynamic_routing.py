'''@file dynamic_routing.py
contains the DynamicRouting decoder'''

import tensorflow as tf
from nabu.neuralnetworks.models.ed_decoders import rnn_decoder
from nabu.neuralnetworks.models.ed_encoders.ed_encoder_factory import factory
from nabu.neuralnetworks.components import rnn_cell as rnn
from nabu.neuralnetworks.components import attention

class DynamicRouting(rnn_decoder.RNNDecoder):
    '''a speller decoder for the LAS architecture'''

    def __init__(self, conf, trainlabels, outputs, constraint, name=None):
        '''EDDecoder constructor

        Args:
            conf: the decoder configuration as a ConfigParser
            trainlabels: the number of extra labels required by the trainer
            outputs: the name of the outputs of the model
            constraint: the constraint for the variables
        '''


        super(DynamicRouting, self).__init__(
            conf, trainlabels, outputs, constraint, name)

        #create the reconstructor
        conf.remove_section('encoder')
        conf.add_section('encoder')
        for option, value in conf.items('reconstruction'):
            conf.set('encoder', option, value)
        conf.remove_section('reconstruction')
        self.reconstructor = factory(conf.get('encoder', 'encoder'))(
            conf, None, 'reconstruction')

    def _decode(self, encoded, encoded_seq_length, targets, target_seq_length,
                is_training):

        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a list of
                [batch_size x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a list of [batch_size] vectors
            targets: the targets used as decoder inputs as a list of
                [batch_size x ...] tensors
            target_seq_length: the sequence lengths of the targets
                as a list of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the decoder as a list of
                [batch_size x ...] tensors
            - the logit sequence_lengths as a list of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

        #get the batch size
        batch_size = tf.shape(targets.values()[0])[0]
        output_dim = self.output_dims.values()[0]
        output_name = self.output_dims.keys()[0]

        #prepend a sequence border label to the targets to get the encoder
        #inputs
        expanded_targets = tf.pad(targets.values()[0], [[0, 0], [1, 0]],
                                  constant_values=output_dim-1)

        #create the rnn cell
        rnn_cell = self.create_cell(encoded, encoded_seq_length, is_training)

        #create the embedding
        embedding = lambda ids: tf.one_hot(
            ids,
            output_dim,
            dtype=tf.float32)

        #create the decoder helper
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=embedding(expanded_targets),
            sequence_length=target_seq_length.values()[0]+1,
            embedding=embedding,
            sampling_probability=float(self.conf['sample_prob'])
        )

        #create the decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=rnn_cell,
            helper=helper,
            initial_state=rnn_cell.zero_state(batch_size, tf.float32)
        )

        #use the decoder
        logits, state, logit_seq_length = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            impute_finished=True)
        logits = logits.rnn_output

        #reconstruct the inputs
        reconstruct = self.conf['to_reconstruct']
        if is_training and float(self.conf['reconstruction_weight']):
            reconstruction = tf.reshape(
                state.reconstruction,
                [batch_size, -1, int(self.conf['capsule_dim'])])
            reconstruction, _ = self.reconstructor(
                {'reconstruction':reconstruction},
                {'reconstruction':encoded_seq_length[reconstruct]},
                is_training)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                (encoded[reconstruct] -  reconstruction['reconstruction'])**2,
                1))
            reconstruction_loss *= float(self.conf['reconstruction_weight'])
            tf.add_to_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES,
                reconstruction_loss)

        return (
            {output_name: logits},
            {output_name: logit_seq_length},
            state)

    def create_cell(self, encoded, encoded_seq_length, is_training):
        '''create the rnn cell

        Args:
            encoded: the encoded sequences as a [batch_size x max_time x dim]
                tensor that will be queried with attention
                set to None if the rnn_cell should be created without the
                attention part (for zero_state)
            encoded_seq_length: the encoded sequence lengths as a [batch_size]
                vector
            is_training: bool whether or not the network is in training mode

        Returns:
            an RNNCell object'''

        decode = self.conf['to_decode']

        rnn_cells = []

        for _ in range(int(self.conf['num_layers'])):

            #create the multilayered rnn cell
            rnn_cell = tf.contrib.rnn.LSTMCell(
                num_units=int(self.conf['num_units']),
                reuse=tf.get_variable_scope().reuse)

            rnn_cells.append(rnn_cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

        attention_mechanism = attention.factory(
            conf=self.conf,
            num_units=rnn_cell.output_size,
            encoded=encoded[decode],
            encoded_seq_length=encoded_seq_length[decode]
        )

        #add attention to the rnn cell
        rnn_cell = rnn.DynamicRoutingAttentionWrapper(
            cell=rnn_cell,
            num_capsules=self.output_dims.values()[0],
            capsule_dim=int(self.conf['capsule_dim']),
            numiters=int(self.conf['routing_iters']),
            attention_mechanism=attention_mechanism,
            input_context=self.conf['input_context'] == 'True',
            input_activation=self.conf['input_activation'] == 'True',
            input_inputs=self.conf['input_inputs'] == 'True'
        )

        rnn_cell = rnn.NormOutputWrapper(rnn_cell)

        return rnn_cell

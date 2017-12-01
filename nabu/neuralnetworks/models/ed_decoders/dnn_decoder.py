'''@ file dnn_decoder.py
contains the DNNDecoder class'''

import tensorflow as tf
import ed_decoder

class DNNDecoder(ed_decoder.EDDecoder):
    '''a DNN decoder'''

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

        #apply for each phonological feature
        outputs = {}
        output_seq_length = {}
        for o in self.output_dims:
            with tf.variable_scope(o):
                output = encoded.values()[0]
                for l in range(int(self.conf['num_layers'])):
                    output = tf.contrib.layers.fully_connected(
                        inputs=output,
                        num_outputs=int(self.conf['num_units']),
                        scope='layer%d' % l
                    )
                    if self.conf['layer_norm'] == 'True':
                        output = tf.contrib.layers.layer_norm(output)

                    if float(self.conf['dropout']) < 1 and is_training:
                        output = tf.nn.dropout(
                            output, float(self.conf['dropout']))

                output = tf.contrib.layers.linear(
                    inputs=output,
                    num_outputs=self.output_dims[o],
                    scope='outlayer'
                )

            outputs[o] = output
            output_seq_length[o] = encoded_seq_length.values()[0]

        return outputs, output_seq_length, ()

    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded sequence as a dictionary
                of integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors'''

        return ()

    def get_output_dims(self, trainlabels):
        '''get the decoder output dimensions

        args:
            targetconfs: the target data confs
            trainlabels: the number of extra labels the trainer needs

        returns:
            a dictionary containing the output dimensions'''

        #get the dimensions of all the targets
        output_dims = {}
        for i, d in enumerate(self.conf['output_dims'].split(' ')):
            output_dims[self.outputs[i]] = int(d) + trainlabels

        return output_dims

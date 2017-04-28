'''@file ctc_decoder.py
contains the CTCDecoder'''

import tensorflow as tf
import decoder

class CTCDecoder(decoder.Decoder):
    '''CTC Decoder'''

    def __call__(self, inputs, input_seq_length):
        '''decode a batch of data

        Args:
            inputs: the inputs as a list of [batch_size x ...] tensors
            input_seq_length: the input sequence lengths as a list of
                [batch_size] vectors

        Returns:
            - the decoded sequences as a list of length beam_width
                containing [batch_size x ...] sparse tensors, the beam elements
                are sorted from best to worst
            - the sequence scores as a [batch_size x beam_width] tensor
        '''

        with tf.name_scope('ctc_decoder'):

            #create the decoding graph
            logits, logits_seq_length =\
                self.model(
                    inputs, input_seq_length, targets=[],
                    target_seq_length=[], is_training=False)

            #Convert logits to time major
            logits = tf.transpose(logits[0], [1, 0, 2])

            #do the CTC beam search
            outputs, logprobs = tf.nn.ctc_greedy_decoder(
                logits, logits_seq_length[0])
            outputs[0] = tf.cast(outputs[0], tf.int32)

        return outputs, logprobs

    @staticmethod
    def get_output_dims(output_dims):
        '''
        Adjust the output dimensions of the model (blank label, eos...)

        Args:
            a list containing the original model output dimensions

        Returns:
            a list containing the new model output dimensions
        '''

        return [output_dim + 1 for output_dim in output_dims]

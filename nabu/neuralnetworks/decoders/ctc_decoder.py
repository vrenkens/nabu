'''@file ctc_decoder.py
contains the CTCDecoder'''

import tensorflow as tf
import decoder
from nabu.processing import score

class CTCDecoder(decoder.Decoder):
    '''CTC Decoder'''

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

        #create the decoding graph
        logits, logits_seq_length =\
            classifier(
                inputs, input_seq_length, targets=None,
                target_seq_length=None, is_training=False)

        #Convert logits to time major
        logits = tf.transpose(logits, [1, 0, 2])

        #do the CTC beam search
        sparse_outputs, logprobs = tf.nn.ctc_greedy_decoder(
            logits, logits_seq_length)
        sparse_outputs = sparse_outputs[0]
        logprobs = tf.unpack(tf.reshape(logprobs, [-1]))

        #split the sparse tensors into the seperate utterances
        output_list = tf.sparse_split(0, self.batch_size, sparse_outputs)
        outputs = [tf.reshape(tf.sparse_tensor_to_dense(o), [-1])
                   for o in output_list]

        outputs = [[(logprobs[i], outputs[i])]
                   for i in range(self.batch_size)]

        return outputs

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''

        #decode the targets
        for utt in targets:
            targets[utt] = self.coder.decode(targets[utt])

        return score.cer(outputs, targets)

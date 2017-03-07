'''@file lm_confidence_decoder.py
contains the LmConfidenceDecoder'''

import tensorflow as tf
import decoder

class LmConfidenceDecoder(decoder.Decoder):
    '''a decoder that will be used to evaluate the language model, it will
       generate a confidence score for the input utterance'''

    def get_outputs(self, inputs, input_seq_length, classifier):

        '''compute the outputs of the decoder

        Args:
            inputs: The inputs to the network as a
                [batch_size x max_input_length x input_dim] tensor
            input_seq_length: The sequence length of the inputs as a
                [batch_size] vector
            classifier: The classifier object that will be used in decoding

        Returns:
            A list with batch_size elements containing a single element list
            with a pair with the confidence score and the input label sequence
        '''

        #compute the output logits
        logits, logit_seq_length =\
            classifier(
                inputs, input_seq_length, targets=None,
                target_seq_length=None, is_training=False)

        #compute the cross enthropy between the inputs and outputs
        with tf.name_scope('cross_enthropy_loss'):
            output_dim = int(logits.get_shape()[2])

            #put all the tragets on top of each other
            split_targets = tf.unstack(tf.cast(inputs, tf.int32)[:, :, 0])
            split_logits = tf.unstack(logits)
            outputs = []
            for i, target in enumerate(split_targets):
                #only use the real data
                target = target[:input_seq_length[i]]
                logit = split_logits[i][:logit_seq_length[i]]

                #append an end of sequence label
                starget = tf.concat([target, [output_dim-1]], 0)

                #one hot encode the targets
                #pylint: disable=E1101
                starget = tf.one_hot(starget, output_dim)

                #compute the perplexity
                loss = tf.exp(tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logit, labels=starget)))

                outputs.append([(loss, target)])

        return outputs

    def score(self, outputs, targets):
        '''score the performance

        Args:
            outputs: a dictionary containing the decoder outputs
            targets: a dictionary containing the targets

        Returns:
            the score'''

        #get the confidence scores for all utterances
        scores = [o[0][0] for o in outputs.values()]

        #return the average confidence score
        return sum(scores)/len(scores)

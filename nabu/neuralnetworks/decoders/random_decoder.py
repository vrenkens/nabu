'''@file random_decoder.py
contains the RandomDecoder'''

import os
import tensorflow as tf
import decoder
from nabu.neuralnetworks.components import ops

class RandomDecoder(decoder.Decoder):
    '''a decoder that returns a random sample from the output distribution'''

    def __init__(self, conf, model):
        '''
        Decoder constructor

        Args:
            conf: the decoder config
            model: the model that will be used for decoding
        '''

        super(RandomDecoder, self).__init__(conf, model)

        #get the alphabet
        self.alphabet = self.conf['alphabet'].split(' ')

    def __call__(self, inputs, input_seq_length):
        '''decode a batch of data

        Args:
            inputs: the inputs as a dictionary of [batch_size x time x ...]
                tensors
            input_seq_length: the input sequence lengths as a dictionary of
                [batch_size] vectors

        Returns:
            - the decoded sequences as a dictionary of outputs
        '''

        with tf.name_scope('random_decoder'):

            output_name = self.model.output_dims.keys()[0]
            batch_size = tf.shape(inputs.values()[0])[0]

            #the start and end tokens are the final output
            token_val = int(self.model.decoder.output_dims.values()[0]-1)

            #start_tokens: vector with size batch_size
            start_tokens = tf.fill([batch_size], token_val)

            #end token
            end_token = tf.constant(token_val, dtype=tf.int32)

            #encode the inputs [batch_size x output_length x output_dim]
            encoded, encoded_seq_length = self.model.encoder(
                inputs=inputs,
                input_seq_length=input_seq_length,
                is_training=False)

            #Use the scope of the decoder so the rnn_cells get reused
            with tf.variable_scope(self.model.decoder.scope):

                #get the RNN cell
                cell = self.model.decoder.create_cell(
                    encoded,
                    encoded_seq_length,
                    False)

            #get the initial state
            initial_state = cell.zero_state(
                batch_size,
                tf.float32)

            #create the embeddings
            embedding = lambda x: tf.one_hot(
                x,
                self.model.decoder.output_dims.values()[0],
                dtype=tf.float32)

            #create the helper
            helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                embedding,
                start_tokens,
                end_token
            )

            #create the decoder
            sample_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell,
                helper,
                initial_state
            )

            #decode using the decoder
            output, _, lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=sample_decoder,
                maximum_iterations=int(self.conf['max_steps']))

            labels = output.sample_id
            logits = output.rnn_output

            #set the labels exeeding the length to zero
            labels = tf.where(
                tf.sequence_mask(lengths),
                labels,
                tf.zeros_like(labels))

            #get the label log probs
            logprobs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits
            )

            #set the log probs exeeding the sequence length to zero
            logprobs = tf.where(
                tf.sequence_mask(lengths),
                logprobs,
                tf.zeros_like(logprobs))

            #compute the probability for all the sequences
            logprobs = tf.reduce_sum(logprobs, 1)

            return {output_name:(labels, lengths, logprobs)}

    def write(self, outputs, directory, names):
        '''write the output of the decoder to disk

        args:
            outputs: the outputs of the decoder
            directory: the directory where the results should be written
            names: the names of the utterances in outputs
        '''
        o = self.model.output_dims.keys()[0]
        batch_size = outputs[o][0].shape[0]
        with open(os.path.join(directory, o), 'a') as fid:
            for i in range(batch_size):
                output = outputs[o][0][i, :outputs[o][1][i]]
                text = ' '.join([self.alphabet[j] for j in output])
                fid.write('%s %s\n' % (names[i], text))

    def evaluate(self, outputs, references, reference_seq_length):
        '''evaluate the output of the decoder

        args:
            outputs: the outputs of the decoder as a dictionary
            references: the references as a dictionary
            reference_seq_length: the sequence lengths of the references

        Returns:
            the error of the outputs
        '''

        #stack all the logits except the final logits
        stacked_outputs = {
            t:ops.seq2nonseq(outputs[t][0], outputs[t][1])
            for t in outputs}


        #create the stacked targets
        stacked_targets = {
            t:tf.cast(ops.seq2nonseq(references[t],
                                     reference_seq_length[t]), tf.int32)
            for t in references}

        #compute the edit distance
        losses = [
            tf.reduce_mean(tf.reduce_mean(tf.cast(tf.not_equal(
                stacked_outputs[o], stacked_targets[o]), tf.float32)))
            for o in outputs]

        loss = tf.reduce_mean(losses)

        return loss

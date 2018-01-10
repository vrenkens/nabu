'''@file random_decoder.py
contains the RandomDecoder'''

import os
import tensorflow as tf
import decoder
from nabu.neuralnetworks.components.ops import dense_sequence_to_sparse

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

    def update_evaluation_loss(self, loss, outputs, references,
                               reference_seq_length):
        '''update the evaluation loss

        args:
            loss: the current evaluation loss
            outputs: the outputs of the decoder as a dictionary
            references: the references as a dictionary
            reference_seq_length: the sequence lengths of the references

        Returns:
            an op to update the evalution loss
        '''

        #create a variable to hold the total number of reference targets
        num_targets = tf.get_variable(
            name='num_targets',
            shape=[],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        sequences = outputs.values()[0][0]
        lengths = outputs.values()[0][1]

        #convert the references to sparse representations
        sparse_targets = dense_sequence_to_sparse(
            references.values()[0], reference_seq_length.values()[0])

        #convert the best sequences to sparse representations
        sparse_sequences = dense_sequence_to_sparse(
            sequences, lengths-1)

        #compute the edit distance
        errors = tf.reduce_sum(
            tf.edit_distance(sparse_sequences, sparse_targets, normalize=False))

        #compute the number of targets in this batch
        batch_targets = tf.reduce_sum(reference_seq_length.values()[0])

        new_num_targets = num_targets + tf.cast(batch_targets, tf.float32)

        #an operation to update the loss
        update_loss = loss.assign(
            (loss*num_targets + errors)/new_num_targets).op

        #add an operation to update the number of targets
        with tf.control_dependencies([update_loss]):
            update_loss = num_targets.assign(new_num_targets).op

        return update_loss

'''@file new_beam_search_decoder.py
contains the BeamSearchDecoder'''

import os
import tensorflow as tf
import decoder
from nabu.neuralnetworks.components.ops import dense_sequence_to_sparse

class BeamSearchDecoder(decoder.Decoder):
    '''Beam search decoder'''

    def __init__(self, conf, model):
        '''
        Decoder constructor

        Args:
            conf: the decoder config
            model: the model that will be used for decoding
        '''

        super(BeamSearchDecoder, self).__init__(conf, model)

        #get the alphabet
        self.alphabet = self.conf['alphabet'].split(' ')



    def __call__(self, inputs, input_seq_length):
        '''decode a batch of data

        Args:
            inputs: the inputs as a dictionary of [batch_size x ...] tensors
            input_seq_length: the input sequence lengths as a dictionary of
                [batch_size] vectors

        Returns:
            - the decoded sequences as a dictionary of outputs
        '''

        with tf.name_scope('beam_search_decoder'):

            output_name = self.model.output_dims.keys()[0]
            beam_width = int(self.conf['beam_width'])
            batch_size = tf.shape(inputs.values()[0])[0]

            #start_tokens: vector with size batch_size
            start_tokens = tf.zeros([batch_size], dtype=tf.int32)

            #encode the inputs
            encoded, encoded_seq_length = self.model.encoder(
                inputs=inputs,
                input_seq_length=input_seq_length,
                is_training=False)

            #repeat the encoded inputs for all beam elements
            encoded = {
                e:tf.contrib.seq2seq.tile_batch(encoded[e], beam_width)
                for e in encoded}

            encoded_seq_length = {
                e:tf.contrib.seq2seq.tile_batch(encoded_seq_length[e],
                                                beam_width)
                for e in encoded_seq_length}


            #Use the scope of the decoder so the rnn_cells get reused
            with tf.variable_scope(self.model.decoder.scope):

                #get the RNN cell
                cell = self.model.decoder.create_cell(
                    encoded,
                    encoded_seq_length,
                    False)

            #get the initial state
            initial_state = cell.zero_state(
                batch_size*beam_width,
                tf.float32)

            #create the embeddings
            embeddings = lambda x: tf.one_hot(
                x+1,
                self.model.decoder.output_dims.values()[0],
                dtype=tf.float32)

            #Create the beam search decoder
            beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cell,
                embedding=embeddings,
                start_tokens=start_tokens,
                end_token=self.model.decoder.output_dims.values()[0]-1,
                initial_state=initial_state,
                beam_width=beam_width,
                length_penalty_weight=float(self.conf['length_penalty']))


            with tf.variable_scope(self.model.decoder.scope):
                #Decode useing the beamsearch decoder
                output, _, lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=beam_search_decoder,
                    maximum_iterations=int(self.conf['max_steps']))

            sequences = output.predicted_ids
            scores = output.beam_search_decoder_output.scores[:, -1, :]


            return {output_name:(sequences, lengths, scores)}

    def write(self, outputs, directory, names):
        '''write the output of the decoder to disk

        args:
            outputs: the outputs of the decoder as a dictionary
            directory: the directory where the results should be written
            names: the names of the utterances in outputs
        '''
        sequences = outputs.values()[0][0]
        scores = outputs.values()[0][2]
        end = int(self.model.decoder.output_dims.values()[0]-1)

        for i, name in enumerate(names):
            with open(os.path.join(directory, name), 'w') as fid:
                for b in range(sequences.shape[2]):
                    sequence = sequences[i, :, b]
                    #look for the first occurence of a sequence border label
                    e = next(i for i, j in enumerate(sequence) if j == end)
                    text = ' '.join([self.alphabet[s] for s in sequence[:e]])
                    fid.write('%f %s\n' % (scores[i, b], text))


    def evaluate(self, outputs, references, reference_seq_length):
        '''evaluate the output of the decoder

        args:
            outputs: the outputs of the decoder as a dictionary
            references: the references as a dictionary
            reference_seq_length: the sequence lengths of the references

        Returns:
            the error of the outputs
        '''

        sequences = outputs.values()[0][0]
        lengths = outputs.values()[0][1]

        #convert the references to sparse representations
        sparse_targets = dense_sequence_to_sparse(
            references.values()[0], reference_seq_length.values()[0])

        #convert the best sequences to sparse representations
        sparse_sequences = dense_sequence_to_sparse(
            sequences[:, :, 0], lengths[:, 0]-1)

        #compute the edit distance
        loss = tf.reduce_mean(
            tf.edit_distance(sparse_sequences, sparse_targets))

        return loss

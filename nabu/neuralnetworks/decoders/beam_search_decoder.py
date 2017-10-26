'''@file new_beam_search_decoder.py
contains the BeamSearchDecoder'''

import os
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest
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

        #get the alphabet
        self.alphabet = conf['alphabet'].split(' ')

        super(BeamSearchDecoder, self).__init__(conf, model)
    def __call__(self, inputs, input_seq_length):
        with tf.name_scope('new_beam_search_decoder'):

            output_name = self.model.output_dims.keys()[0]
            self.max_output_length = int(self.conf['max_steps'])
            beam_width = int(self.conf['beam_width'])
            batch_size = tf.shape(inputs.values()[0])[0]

            token_val = int(self.model.decoder.output_dims.values()[0]-1)
            #start_tokens:vector with size batch_size
            start_tokens = tf.fill([batch_size],token_val)

            end_token = tf.constant(
                token_val,dtype=tf.int32)

            #encode the inputs [batch_size x output_length x output_dim]
            encoded, encoded_seq_length = self.model.encoder(
                inputs=inputs,
                input_seq_length=input_seq_length,
                is_training=False)

            #repeat the encoded inputs for all beam elements
            encoded = {e:tf.contrib.seq2seq.tile_batch(encoded[e],
                beam_width)
                for e in encoded}
            encoded_seq_length = {e:tf.contrib.seq2seq.tile_batch(
                encoded_seq_length[e],beam_width)
                for e in encoded_seq_length}


            #Use the scope of the decoder so the rnn_cells get reused
            with tf.variable_scope(self.model.decoder.scope):

                #get the RNN cell
                cell = self.model.decoder.create_cell(
                    encoded,
                    encoded_seq_length,
                    False)

                #get the initial state
                initial_state = cell.zero_state(batch_size*beam_width,
                    tf.float32)

                #create the embeddings
                embeddings = lambda x:tf.one_hot(x,
                    self.model.decoder.output_dims.values()[0],dtype=tf.float32)

                #Create the beam search decoder
                beam = tf.contrib.seq2seq.BeamSearchDecoder(cell,
                    embeddings,start_tokens,end_token,initial_state,beam_width,
                    output_layer=None,
                    length_penalty_weight=0.0)


                #Decode useing the beamsearch decoder
                test_output, _, test_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder            = beam,
                    maximum_iterations = self.max_output_length)


            return {output_name:(test_output, test_lengths)}

    def write(self, outputs, directory, names):
        '''write the output of the decoder to disk

        args:
            outputs: the outputs of the decoder as a dictionary
            directory: the directory where the results should be written
            names: the names of the utterances in outputs
        '''
        sequences = outputs.values()[0][0].predicted_ids[:,:,0]
        lengths = outputs.values()[0][1]

        for i, name in enumerate(names):
            with open(os.path.join(directory, name), 'w') as fid:
                for b in range(sequences.shape[0]):
                    sequence = sequences[b, i][:lengths[b, i]]
                    text = ' '.join([self.alphabet[s] for s in sequence])
                    fid.write('%f %s\n' % (text))

    def to_sparse(self, tensor, lengths):
        mask = tf.sequence_mask(lengths, self.max_output_length)
        indices = tf.to_int64(tf.where(tf.equal(mask, True)))
        values = tf.to_int32(tf.boolean_mask(tensor, mask))
        shape = tf.to_int64(tf.shape(tensor))
        return tf.SparseTensor(indices, values, shape)

    def evaluate(self, outputs, references, reference_seq_length):
        '''evaluate the output of the decoder

        args:
            outputs: the outputs of the decoder as a dictionary
            references: the references as a dictionary
            reference_seq_length: the sequence lengths of the references

        Returns:
            the error of the outputs
        '''
        #create sparse representaions of predictions and remove EOS token
        c = outputs.values()[0][0].predicted_ids[:,:,0]
        c = tf.Print(c,[c],"predicted", summarize = 50)
        d = outputs.values()[0][1][:, 0] - 1
        d = tf.Print(d,[d],"predicted-lengths",summarize = 32)
        predicts = dense_sequence_to_sparse(c, d)

        #create sparse representaion of references
        a = reference_seq_length.values()[0]
        a = tf.Print(a,[a],"target-lengths",summarize = 32)
        b = references.values()[0]
        b = tf.Print(b,[b],"targets",summarize = 50)
        labels = dense_sequence_to_sparse(
            b, a)

        #calculate error rate
        error_rate = tf.reduce_mean(tf.edit_distance(predicts, labels))

        return error_rate

def _stack_beam(tensor):
    '''converts a [Batch_size x beam_width x ...] Tensor into a
    [Batch_size * beam_width x ...] Tensor

    Args:
        The [Batch_size x beam_width x ...] Tensor to be converted

    Returns:
        The converted [Batch_size * beam_width x ...] Tensor'''

    return tf.concat(tf.unstack(tensor, axis=1), axis=0)

def _unstack_beam(tensor, beam_width):
    '''converts a [Batch_size * beam_width x ...] Tenor into a
    [Batch_size x beam_width x ...] Tensor

    Args:
        The [Batch_size * beam_width x ...] Tensor to be converted
        batch_size: the batch size

    Returns:
        The converted [Batch_size x beam_width x ...] Tensor'''

    return tf.stack(tf.split(tensor, beam_width), axis=1)

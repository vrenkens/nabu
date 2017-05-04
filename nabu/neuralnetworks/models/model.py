'''@file model.py
contains de Model class'''

from ed_encoders import ed_encoder_factory
from ed_decoders import ed_decoder_factory

class Model(object):
    '''a general class for an encoder decoder system'''

    def __init__(self, conf, output_dims):
        '''Model constructor

        Args:
            conf: The model configuration as a configparser object
            output_dims: a list containing the model output dimensions
            name: the model name
        '''

        #create the encoder
        encoder_conf = dict(conf.items('encoder'))
        self.encoder = ed_encoder_factory.factory(encoder_conf['encoder'])(
            encoder_conf)

        #create the decoder
        decoder_conf = dict(conf.items('decoder'))
        self.decoder = ed_decoder_factory.factory(decoder_conf['decoder'])(
            decoder_conf, output_dims)

        self.output_dims = output_dims

        self.input_names = conf.get('io', 'inputs').split(' ')
        if self.input_names == ['']:
            self.input_names = []
        self.output_names = conf.get('io', 'outputs').split(' ')
        if self.output_names == ['']:
            self.output_names = []

    def __call__(self, inputs, input_seq_length, targets,
                 target_seq_length, is_training):

        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a list of
                [batch_size x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a list of [batch_size] vectors
            targets: the targets to the neural network, this is a list of
                [batch_size x max_output_length] tensors.
            target_seq_length: The sequence lengths of the target utterances,
                this is a list of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - output logits, which is a list of [batch_size x ...] tensors
            - the output logits sequence lengths which is a list of [batch_size]
                vectors
        '''

        #compute the high level features
        encoded, encoded_seq_length = self.encoder(
            inputs=inputs,
            input_seq_length=input_seq_length,
            is_training=is_training)

        #compute the output logits
        logits, logit_seq_length, _ = self.decoder(
            encoded=encoded,
            encoded_seq_length=encoded_seq_length,
            targets=targets,
            target_seq_length=target_seq_length,
            is_training=is_training)

        return logits, logit_seq_length

    @property
    def variables(self):
        '''get a list of the models's variables'''

        return self.encoder.variables + self.decoder.variables

'''@file parallel_encoder.py
contains the ParallelEncoder'''

import StringIO
import ConfigParser
import ed_encoder
import ed_encoder_factory

class ParallelEncoder(ed_encoder.EDEncoder):
    '''this encoder will use encoders in parallel'''

    def __init__(self, conf, constraint, name=None):
        '''constructor

        Args:
            conf: the encoder configuration
            name: the encoder name
            constraint: the constraint for the variables
        '''


        super(ParallelEncoder, self).__init__(conf, constraint, name)

        #create the encoders
        self.encoders = {}
        for encoder in self.conf['encoders'].split():
            # Create a deep copy of the conf
            config_string = StringIO.StringIO()
            conf.write(config_string)

            # We must reset the buffer to make it ready for reading.
            config_string.seek(0)
            encoder_conf = ConfigParser.ConfigParser()
            encoder_conf.readfp(config_string)

            encoder_conf.remove_section('encoder')
            encoder_conf.add_section('encoder')

            #set the wrapped section as the trainer section
            for option, value in conf.items(encoder):
                encoder_conf.set('encoder', option, value)

            #remove the other encoder sections
            for remove in self.conf['encoders'].split():
                encoder_conf.remove_section(remove)

            self.encoders[encoder] = ed_encoder_factory.factory(
                encoder_conf.get('encoder', 'encoder'))(
                    encoder_conf, constraint, encoder
                )

    def encode(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            is_training: whether or not the network is in training mode

        Returns:
            - the outputs of the encoder as a dictionary of
                [bath_size x time x ...] tensors
            - the sequence lengths of the outputs as a dictionary of
                [batch_size] tensors
        '''

        #call the wrapped encoder
        encoded, encoded_seq_length = {}, {}
        for encoder in self.encoders:
            enc, lengths = self.encoders[encoder](
                inputs, input_seq_length, is_training)
            for e in enc:
                name = e + '_' + encoder
                encoded[name] = enc[e]
                encoded_seq_length[name] = lengths[e]

        return encoded, encoded_seq_length

'''@dummy_encoder.py
contains the DummyEncoder class'''

import ed_encoder

class DummyEncoder(ed_encoder.EDEncoder):
    '''an encoder that does nothing'''

    def encode(self, inputs, input_seq_length, is_training=False):
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

        return inputs, input_seq_length

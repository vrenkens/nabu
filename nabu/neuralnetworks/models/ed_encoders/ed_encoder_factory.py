'''@file ed_encoder_factory.py
contains the EDEncoder factory'''

def factory(encoder):
    '''get an EDEncoder class

    Args:
        encoder: the encoder type

    Returns:
        an EDEncoder class'''

    if encoder == 'listener':
        import listener
        return listener.Listener
    elif encoder == 'dummy_encoder':
        import dummy_encoder
        return dummy_encoder.DummyEncoder
    elif encoder == 'dblstm':
        import dblstm
        return dblstm.DBLSTM
    elif encoder == 'dnn':
        import dnn
        return dnn.DNN
    elif encoder == 'hotstart_encoder':
        import hotstart_encoder
        return hotstart_encoder.HotstartEncoder
    else:
        raise Exception('undefined encoder type: %s' % encoder)

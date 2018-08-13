'''@file tfreader_factory.py
contains the tfreader factory'''

def factory(datatype):
    '''factory for tfreaders

    Args:
        dataype: the type of data to be read

    Returns:
        a tfreader class
    '''

    if datatype == 'audio_feature':
        import audio_feature_reader
        return audio_feature_reader.AudioFeatureReader
    elif datatype == 'string':
        import string_reader
        return string_reader.StringReader
    elif datatype == 'string_eos':
        import string_reader_eos
        return string_reader_eos.StringReaderEOS
    elif datatype == 'binary':
        import binary_reader
        return binary_reader.BinaryReader
    elif datatype == 'alignment':
        import alignment_reader
        return alignment_reader.AlignmentReader
    else:
        raise Exception('unknown data type: %s' % datatype)

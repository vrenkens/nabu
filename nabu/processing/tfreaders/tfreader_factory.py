'''@file tfreader_factory.py
contains the tfreader factory'''

from . import audio_feature_reader, string_reader, binary_reader, \
alignment_reader

def factory(datatype):
    '''factory for tfreaders

    Args:
        dataype: the type of data to be read

    Returns:
        a tfreader class
    '''

    if datatype == 'audio_feature':
        return audio_feature_reader.AudioFeatureReader
    elif datatype == 'string':
        return string_reader.StringReader
    elif datatype == 'binary':
        return binary_reader.BinaryReader
    elif datatype == 'alignment':
        return alignment_reader.AlignmentReader
    else:
        raise Exception('unknown data type: %s' % datatype)

'''@file tfwriter_factory
contains the tfwriter factory'''

from . import array_writer, string_writer, binary_writer, alignment_writer

def factory(datatype):
    '''
    Args:
        datatype: the type of data to be written

    Returns:
        a tfwriter class
    '''

    if datatype == 'audio_feature':
        return array_writer.ArrayWriter
    elif datatype == 'string':
        return string_writer.StringWriter
    elif datatype == 'binary':
        return binary_writer.BinaryWriter
    elif datatype == 'alignment':
        return alignment_writer.AlignmentWriter
    else:
        raise Exception('unknown data type: %s' % datatype)

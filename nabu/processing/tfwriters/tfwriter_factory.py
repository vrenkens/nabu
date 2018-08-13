'''@file tfwriter_factory
contains the tfwriter factory'''

def factory(datatype):
    '''
    Args:
        datatype: the type of data to be written

    Returns:
        a tfwriter class
    '''

    if datatype == 'audio_feature':
        import array_writer
        return array_writer.ArrayWriter
    elif datatype == 'string' or datatype == 'string_eos':
        import string_writer
        return string_writer.StringWriter
    elif datatype == 'binary':
        import binary_writer
        return binary_writer.BinaryWriter
    elif datatype == 'alignment':
        import alignment_writer
        return alignment_writer.AlignmentWriter
    else:
        raise Exception('unknown data type: %s' % datatype)

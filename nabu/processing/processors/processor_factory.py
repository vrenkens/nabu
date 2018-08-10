'''@file processor_factory.py
contains the Processor factory method'''

def factory(processor):
    '''gets a Processor class

    Args:
        processor: the processor type

    Returns:
        a Processor class'''

    if processor == 'audio_processor':
        import audio_processor
        return audio_processor.AudioProcessor
    elif processor == 'text_processor':
        import text_processor
        return text_processor.TextProcessor
    elif processor == 'binary_processor':
        import binary_processor
        return binary_processor.BinaryProcessor
    elif processor == 'alignment_processor':
        import alignment_processor
        return alignment_processor.AlignmentProcessor
    elif processor == 'textfile_processor':
        import textfile_processor
        return textfile_processor.TextFileProcessor
    else:
        raise Exception('unknown processor type: %s' % processor)

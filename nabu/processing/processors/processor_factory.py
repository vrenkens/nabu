'''@file processor_factory.py
contains the Processor factory method'''

from . import audio_processor, text_processor

def factory(processor):
    '''gets a Processor class

    Args:
        processor: the processor type

    Returns:
        a Processor class'''

    if processor == 'audio_processor':
        return audio_processor.AudioProcessor
    elif processor == 'text_processor':
        return text_processor.TextProcessor
    else:
        raise Exception('unknown processor type: %s' % processor)

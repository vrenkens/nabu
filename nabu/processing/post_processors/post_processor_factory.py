'''@file post_processor_factory.py
contains the PostProcessor factory method'''

from . import text_post_processor

def factory(post_processor):
    '''gets a Processor class

    Args:
        post_processor: the post processor type

    Returns:
        a PostProcessor class'''

    if post_processor == 'text_post_processor':
        return text_post_processor.TextPostProcessor
    else:
        raise Exception('unknown post processor type: %s' % post_processor)

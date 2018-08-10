'''@file processor.py
contains the Processor class'''

import os
from abc import ABCMeta, abstractmethod
from nabu.tools.default_conf import apply_defaults

class Processor(object):
    '''general Processor class for data processing'''

    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''Processor constructor

        Args:
            conf: processor configuration as a configparser
        '''

        self.conf = dict(conf.items('processor'))

        #apply default configuration
        default = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'defaults',
            type(self).__name__.lower() + '.cfg')
        apply_defaults(self.conf, default)

    @abstractmethod
    def __call__(self, dataline):
        '''process the data in dataline
        Args:
            dataline: a string, can be a line of text a pointer to a file etc.

        Returns:
            The processed data'''

    @abstractmethod
    def write_metadata(self, datadir):
        '''write the processor metadata to disk

        Args:
            dir: the directory where the metadata should be written'''

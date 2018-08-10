'''@file processor.py
contains the Processor class'''

from abc import ABCMeta, abstractmethod

class Processor(object):
    '''general Processor class for data processing'''

    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''Processor constructor

        Args:
            conf: processor configuration as a configparser
        '''

        self.conf = dict(conf.items('processor'))

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

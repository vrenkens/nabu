'''@file tfreader.py
contains the TfReader class'''

from abc import ABCMeta, abstractmethod, abstractproperty
import tensorflow as tf

class TfReader(object):
    '''class for reading tfrecord files and processing them'''

    __metaclass__ = ABCMeta

    def __init__(self, datadirs):
        '''TfReader constructor

        Args:
            datadirs: the directories where the metadata was stored as a list
                of strings
        '''

        #read the metadata
        self.metadata = self._read_metadata(datadirs)

        #create the features object
        self.features = self._create_features()

        #create a reader
        self.reader = tf.TFRecordReader()


    def __call__(self, queue, name=None):
        '''read all data from the queue

        Args:
            queue: a queue containing filenames of tf record files
            name: the name of the operation

        Returns:
            a pair of tensor and sequence length
        '''
        with tf.name_scope(name or type(self).__name__):

            #read all the elements in the queue
            _, serialized = self.reader.read(queue)

            #parse the serialized strings into features
            features = tf.parse_single_example(serialized, self.features)

            #process the parsed features
            processed = self._process_features(features)

        return processed

    @abstractmethod
    def _read_metadata(self, datadirs):
        '''read the metadata for the reader (writen by the processor)

            Args:
                datadirs: the directories where the metadata was stored as a
                    list of strings

            Returns:
                the metadata as a dictionary
        '''

    @abstractmethod
    def _create_features(self):
        '''
            creates the information about the features

            Returns:
                A dict mapping feature keys to FixedLenFeature, VarLenFeature,
                and SparseFeature values
        '''

    @abstractmethod
    def _process_features(self, features):
        '''process the read features

        features:
            A dict mapping feature keys to Tensor and SparseTensor values

        Returns:
            a pair of tensor and sequence length
        '''

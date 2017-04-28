'''@file string_reader.py
contains the StringReader class'''

import os
import numpy as np
import tensorflow as tf
import tfreader

class StringReader(tfreader.TfReader):
    '''a reader for reading and encoding text data'''

    def _read_metadata(self, datadir):
        '''read the metadata for the reader (writen by the processor)

            Args:
                datadir: the directory where the metadata was written

            Returns:
                the metadata as a dictionary
        '''

        metadata = dict()

        #read the maximum length
        with open(os.path.join(datadir, 'max_length')) as fid:
            metadata['max_length'] = int(fid.read())

        #read the sequence length histogram
        with open(os.path.join(datadir,
                               'sequence_length_histogram.npy')) as fid:
            metadata['sequence_length_histogram'] = np.load(fid)

        #read the alphabet
        with open(os.path.join(datadir, 'alphabet')) as fid:
            alphabet = fid.read().split()
        metadata['alphabet'] = tf.constant(alphabet)

        return metadata

    def _create_features(self):
        '''
            creates the information about the features

            Returns:
                A dict mapping feature keys to FixedLenFeature, VarLenFeature,
                and SparseFeature values
        '''
        return {
            'length':tf.FixedLenFeature(
                shape=[],
                dtype=tf.int64,
                default_value=0),
            'data': tf.FixedLenFeature(dtype=tf.string, shape=[])}

    def _process_features(self, features):
        '''process the read features

        features:
            A dict mapping feature keys to Tensor and SparseTensor values

        Returns:
            a pair of tensor and sequence length
        '''

        #split the data string
        splitstring = tf.reshape(tf.sparse_tensor_to_dense(
            tf.string_split([features['data']], ' '), ''), [-1])

        #encode the string by looking up the characters in the alphabet
        data = tf.where(tf.equal(
            tf.expand_dims(splitstring, 1),
            self.metadata['alphabet']))[:, 1]
        sequence_length = tf.shape(data)[0]
        data = tf.cast(data, tf.int32)

        #pad the data untill the maximal length if required
        paddings = [[0, self.metadata['max_length'] - sequence_length]]
        data = tf.pad(data, paddings)
        data.set_shape([self.metadata['max_length']])

        return data, sequence_length

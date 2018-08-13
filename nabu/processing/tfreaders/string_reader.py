'''@file string_reader.py
contains the StringReader class'''

import os
import numpy as np
import tensorflow as tf
import tfreader

class StringReader(tfreader.TfReader):
    '''a reader for reading and encoding text data'''

    def _read_metadata(self, datadirs):
        '''read the metadata for the reader (writen by the processor)

            Args:
                datadirs: the directories where the metadata was stored as a
                    list of strings

            Returns:
                the metadata as a dictionary
        '''

        metadata = dict()

        #read the maximum lengths
        max_lengths = []
        for datadir in datadirs:
            with open(os.path.join(datadir, 'max_length')) as fid:
                max_lengths.append(int(fid.read()))
        metadata['max_length'] = max(max_lengths)

        #read the sequence length histograms
        metadata['sequence_length_histogram'] = np.zeros(
            [metadata['max_length'] + 1])
        for datadir in datadirs:
            with open(os.path.join(datadir,
                                   'sequence_length_histogram.npy')) as fid:
                histogram = np.load(fid)
                metadata['sequence_length_histogram'][:histogram.shape[0]] += (
                    histogram
                )

        with open(os.path.join(datadirs[0], 'nonesymbol')) as fid:
            nonesymbol = fid.read()

        #read the alphabets
        with open(os.path.join(datadirs[0], 'alphabet')) as fid:
            alphabet = fid.read().split()
        for datadir in datadirs:
            with open(os.path.join(datadir, 'alphabet')) as fid:
                if alphabet != fid.read().split():
                    raise Exception(
                        'all string reader alphabets must be the same')

        alphabet = [nonesymbol] + alphabet

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
        data = tf.cast(data, tf.int32) - 1

        assert_op = tf.assert_equal(
            tf.shape(splitstring)[0], sequence_length,
            data=[features['data']],
            message='not all string elements found in alphabet')

        with tf.control_dependencies([assert_op]):
            data = tf.identity(data)

        return data, sequence_length

'''@file audio_feature_reader.py
contains the AudioFeatureReader class'''

import os
import numpy as np
import tensorflow as tf
import tfreader

class AudioFeatureReader(tfreader.TfReader):
    '''reader for audio features'''

    def _read_metadata(self, datadirs):
        '''read the mean and std for normalization and the input dimension

            Args:
                datadir: the directory where the metadata was written

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

        #read the dimension of the data
        with open(os.path.join(datadirs[0], 'dim')) as fid:
            metadata['dim'] = int(fid.read())
        for datadir in datadirs:
            with open(os.path.join(datadir, 'dim')) as fid:
                if metadata['dim'] != int(fid.read()):
                    raise Exception(
                        'all audio feature reader dimensions must be the same')

        return metadata

    def _create_features(self):
        '''
            creates the information about the features

            Returns:
                A dict mapping feature keys to FixedLenFeature, VarLenFeature,
                and SparseFeature values
        '''

        return {'data': tf.FixedLenFeature([], dtype=tf.string)}

    def _process_features(self, features):
        '''process the read features

        features:
            A dict mapping feature keys to Tensor and SparseTensor values

        Returns:
            a pair of tensor and sequence length
        '''

        data = tf.decode_raw(features['data'], tf.float32)
        data = tf.reshape(data, [-1, self.metadata['dim']])
        sequence_length = tf.shape(data)[0]

        return data, sequence_length

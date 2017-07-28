'''@file alignment_reader.py
contains the AlignmentReader class'''

import os
import numpy as np
import tensorflow as tf
import tfreader

class AlignmentReader(tfreader.TfReader):
    '''reader for kaldi alignments'''

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

        data = tf.decode_raw(features['data'], tf.int32)
        sequence_length = tf.shape(data)[0]

        return data, sequence_length

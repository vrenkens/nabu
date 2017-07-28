'''@file alignment_writer.py
contains the AlignmentWriter class'''

import numpy as np
import tensorflow as tf
import tfwriter

class AlignmentWriter(tfwriter.TfWriter):
    '''a TfWriter to write kaldi alignments'''

    def _get_example(self, data):
        '''write data to a file

        Args:
            data: the data to be written'''

        data_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[data.reshape([-1]).astype(np.int32).tostring()]))

        #create the example proto
        example = tf.train.Example(features=tf.train.Features(feature={
            'data': data_feature}))

        return example

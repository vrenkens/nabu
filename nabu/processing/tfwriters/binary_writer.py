'''@file binary_writer.py
contains the BinaryWriter class'''

import tensorflow as tf
import tfwriter

class BinaryWriter(tfwriter.TfWriter):
    '''a TfWriter to write strings'''

    def _get_example(self, data):
        '''write data to a file

        Args:
            data: the data to be written
        '''

        length_feature = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[data.shape[0]]))
        data_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[data.tostring()]))

        #create the example proto
        example = tf.train.Example(features=tf.train.Features(feature={
            'data': data_feature}))

        return example

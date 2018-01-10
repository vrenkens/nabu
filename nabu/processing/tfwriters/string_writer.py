'''@file string_writer.py
contains the StringWriter class'''

import tensorflow as tf
import tfwriter

class StringWriter(tfwriter.TfWriter):
    '''a TfWriter to write strings'''

    def _get_example(self, data):
        '''write data to a file

        Args:
            data: the data to be written
        '''

        length_feature = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[len(data)]))
        data_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[data]))

        #create the example proto
        example = tf.train.Example(features=tf.train.Features(feature={
            'length': length_feature,
            'data': data_feature}))

        return example

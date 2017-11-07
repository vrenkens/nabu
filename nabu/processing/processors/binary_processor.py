'''@file binary_processor.py
Contains the BinaryProcessor'''

import os
import numpy as np
import processor

class BinaryProcessor(processor.Processor):
    '''a processor for text data, does normalization'''

    def __init__(self, conf):
        '''TextProcessor constructor

        Args:
            conf: processor configuration as a configparser
        '''

        #initialize the metadata
        self.max_length = 0
        self.sequence_length_histogram = np.zeros(0, dtype=np.int32)

        super(BinaryProcessor, self).__init__(conf)

    def __call__(self, dataline):
        '''process the data in dataline
        Args:
            dataline: a line of text

        Returns:
            The binary data as an array'''

        split = dataline.split(' ')
        seq_length = len(split)
        binary = np.array(map(int, split)).astype(np.uint8)

        if self.conf['max_length'] != 'None':
            max_length = int(self.conf['max_length'])
        else:
            max_length = None

        if not max_length or seq_length <= max_length:

            #update the metadata
            self.max_length = max(self.max_length, seq_length)
            if seq_length >= self.sequence_length_histogram.shape[0]:
                self.sequence_length_histogram = np.concatenate(
                    [self.sequence_length_histogram, np.zeros(
                        seq_length-self.sequence_length_histogram.shape[0]+1,
                        dtype=np.int32)]
                )
            self.sequence_length_histogram[seq_length] += 1

            return binary

        else:
            return None

    def write_metadata(self, datadir):
        '''write the processor metadata to disk

        Args:
            dir: the directory where the metadata should be written'''

        with open(os.path.join(datadir, 'max_length'), 'w') as fid:
            fid.write(str(self.max_length))
        with open(os.path.join(datadir, 'sequence_length_histogram.npy'),
                  'w') as fid:
            np.save(fid, self.sequence_length_histogram)

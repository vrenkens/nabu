'''@file alignment_processor.py
Contains the TextProcessor'''

import os
import numpy as np
import processor

class AlignmentProcessor(processor.Processor):
    '''a processor for kaldi alignments'''

    def __init__(self, conf):
        '''AlignmentProcessor constructor

        Args:
            conf: the textprocessor configuration as a dict of strings'''


        #initialize the metadata
        self.max_length = 0
        self.sequence_length_histogram = np.zeros(0, dtype=np.int32)
        self.dim = 0

        super(AlignmentProcessor, self).__init__(conf)

    def __call__(self, dataline):
        '''process the data in dataline

        Args:
            dataline: a line containing alignments

        Returns:
            A numpy array with the alignments'''

        #normalize the line
        alignments = np.array(map(int, dataline.split()))

        #update the metadata
        seq_length = alignments.size
        self.max_length = max(self.max_length, seq_length)
        if seq_length >= self.sequence_length_histogram.shape[0]:
            self.sequence_length_histogram = np.concatenate(
                [self.sequence_length_histogram, np.zeros(
                    seq_length-self.sequence_length_histogram.shape[0]+1,
                    dtype=np.int32)]
            )
        self.sequence_length_histogram[seq_length] += 1
        self.dim = max(self.dim, alignments.max())

        return alignments

    def write_metadata(self, datadir):
        '''write the processor metadata to disk

        Args:
            dir: the directory where the metadata should be written'''

        with open(os.path.join(datadir, 'max_length'), 'w') as fid:
            fid.write(str(self.max_length))
        with open(os.path.join(datadir, 'sequence_length_histogram.npy'),
                  'w') as fid:
            np.save(fid, self.sequence_length_histogram)
        with open(os.path.join(datadir, 'dim'), 'w') as fid:
            fid.write(str(self.dim+1))

'''@file text_post_processor.py
contains the TextPostProcessor class'''

import os
from post_processor import PostProcessor
from nabu.processing.target_normalizers import normalizer_factory

class TextPostProcessor(PostProcessor):
    '''a post processor for text data'''

    def __init__(self, conf):
        '''PostProcessor constructor

        Args:
            conf: post processor configuration'''

        #create the normalizer
        normalizer = normalizer_factory.factory(conf['normalizer'])()

        #get the alphabet
        self.alphabet = normalizer.alphabet

        super(TextPostProcessor, self).__init__(conf)

    def process(self, output):
        '''process a label sequence

        Args:
            output: the output of a single example as a numpy array

        Returns:
            The processed sequence
        '''
        return ' '.join([self.alphabet[i] for i in output])

    def write(self, processed, directory, name):
        '''write the processed data to disk

        Args:
            processed: the processed data as a list of length beam width
                containing pairs of score and processed sequences
            directory: the directory to write to
            name: the id of the data
        '''

        with open(os.path.join(directory, name), 'w') as fid:
            for p in processed:
                fid.write('%f\t%s\n' % p)

    @property
    def num_labels(self):
        '''the number of possibke labels as an int'''

        return len(self.alphabet)

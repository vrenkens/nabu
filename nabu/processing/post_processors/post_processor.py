'''@file post_processor.py
contains the PostProcessor class'''

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class PostProcessor(object):
    '''a PostProcessor is used to do postprocessing on the decoder outputs'''

    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''PostProcessor constructor

        Args:
            conf post processor configuration'''

        self.conf = conf

    def __call__(self, outputs, scores):
        '''process a batch of data

        - the decoded sequences as a list of length beam_width
            containing [batch_size x ...] SparseTensorValue objects,
            the beam elements are sorted from best to worst
        - the sequence scores as a [batch_size x beam_width] numpy array

        Returns:
            a list of length batch size containing a list of length beam width
            containing pairs of score and processed'''

        beam_width = len(outputs)
        batch_size = outputs[0].dense_shape[0]

        processed = []
        for i in range(batch_size):
            processed.append([])
            for j in range(beam_width):
                #convert the sparse tensor sequence to an array of labels
                indices = np.where(outputs[j].indices[:, 0] == i)[0]
                output = outputs[j].values[indices]
                processed[i].append((scores[i, j], self.process(output)))

        return processed

    @abstractmethod
    def process(self, output):
        '''process a label sequence

        Args:
            output: the output of a single example as a numpy array

        Returns:
            The processed sequence
        '''

    @abstractmethod
    def write(self, processed, directory, name):
        '''write the processed data to disk

        Args:
            processed: the processed data as a list of length beam width
                containing pairs of score and processed sequences
            directory: the directory to write to
            name: the id of the data
        '''

    @abstractproperty
    def num_labels(self):
        '''the number of possibke labels as an int'''

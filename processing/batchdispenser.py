'''@file batchdispenser.py
contains the functionality for read features and batches of features for neural network training and testing'''

import numpy as np
import readfiles

class Batchdispenser(object):
    '''Class that dispenses batches of data for mini-batch training'''

    def __init__(self, feature_reader, size, alifile, num_labels):
        '''
        Batchdispenser constructor

        Args:
            feature_reader: a feature reader object
            size: the batch size
            scpfile: the path to the features .scp file
            alifile: the path to the file containing the alignments
            num_labels: total number of labels
        '''

        #store the feature reader
        self.feature_reader = feature_reader

        #read the alignments
        self.alignments = readfiles.read_alignments(alifile)

        #save the number of labels
        self.num_labels = num_labels

        #store the batch size
        self.size = size

    def get_batch(self):
        '''
        get a batch of features and alignments in one-hot encoding

        Returns:
            A pair containing:

            -a batch of data
            -the labels in one hot encoding
        '''

        numutt = 0
        batch_data = np.empty(0)
        batch_labels = np.empty(0)

        while numutt < self.size:

            #read utterance
            utt_id, utt_mat, _ = self.feature_reader.get_utt()

            #check if utterance has an alignment
            if utt_id in self.alignments:

                #add the features and alignments to the batch
                batch_data = np.append(batch_data, utt_mat)
                batch_labels = np.append(batch_labels, self.alignments[utt_id])

                #update number of utterances in the batch
                numutt += 1
            else:
                print 'WARNING no alignment for %s' % utt_id

        #reahape the batch data
        batch_data = batch_data.reshape(batch_data.size/utt_mat.shape[1], utt_mat.shape[1])

        #put labels in one hot encoding
        batch_labels = (np.arange(self.num_labels) == batch_labels[:, np.newaxis]).astype(np.float32)

        return (batch_data, batch_labels)

    def split(self):
        '''split of the part that has allready been read by the batchdispenser, this can be used to read a validation set and then split it of from the rest'''

        self.feature_reader.split()

    def skip_batch(self):
        '''skip a batch'''

        numutt = 0
        while numutt < self.size:
            #read utterance
            utt_id = self.feature_reader.next_id()

            #check if utterance has an alignment
            if utt_id in self.alignments:

                #update number of utterances in the batch
                numutt += 1

    def return_batch(self):
        '''return to the previous batch'''

        numutt = 0
        while numutt < self.size:
            #read utterance
            utt_id = self.feature_reader.prev_id()

            #check if utterance has an alignment
            if utt_id in self.alignments:

                #update number of utterances in the batch
                numutt += 1

    def compute_prior(self):
        '''
        compute the pior probability of the labels in alignments

        Returns:
            a numpy array containing the label prior probabilities
        '''

        prior = np.array([(np.arange(self.num_labels) == alignment[:, np.newaxis]).astype(np.float32).sum(0) for alignment in self.alignments.values()]).sum(0)
        return prior/prior.sum()

    @property
    def num_utt(self):
        '''the number of utterances'''

        return len(self.alignments)

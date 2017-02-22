'''@file feature_reader.py
reading features and applying cmvn and splicing them'''

import copy
import ark
import numpy as np
import readfiles

class FeatureReader(object):
    '''Class that can read features from a Kaldi archive and process
    them (cmvn and splicing)'''

    def __init__(self, scpfile, cmvnfile, utt2spkfile, max_length):
        '''
        create a FeatureReader object

        Args:
            scpfile: path to the features .scp file
            cmvnfile: path to the cmvn file
            utt2spkfile:path to the file containing the mapping from utterance
                ID to speaker ID
            max_length: the maximum length of all the utterances in the
                scp file
        '''

        #create the feature reader
        self.reader = ark.ArkReader(scpfile)

        #create a reader for the cmvn statistics
        self.reader_cmvn = ark.ArkReader(cmvnfile)

        #save the utterance to speaker mapping
        self.utt2spk = readfiles.read_utt2spk(utt2spkfile)

        #store the max length
        self.max_length = max_length

    def get_utt(self):
        '''
        read the next features from the archive, normalize and splice them

        Returns:
            the normalized and spliced features
        '''

        #read utterance
        (utt_id, utt_mat, looped) = self.reader.read_next_utt()

        #apply cmvn
        cmvn_stats = self.reader_cmvn.read_utt_data(self.utt2spk[utt_id])
        utt_mat = apply_cmvn(utt_mat, cmvn_stats)

        return utt_id, utt_mat, looped

    def split(self, num_utt):
        '''take a number of utterances from the feature reader to make a new one

        Args:
            num_utt: the number of utterances in the new feature reader

        Returns:
            a feature reader with the requested number of utterances'''
        #create a copy of self
        reader = copy.deepcopy(self)

        #split of a part of the ark reader
        reader.reader = self.reader.split(num_utt)

        return reader

    @property
    def num_utt(self):
        '''number of utterances in the reader'''
        return self.reader.num_utt

    @property
    def pos(self):
        '''the position in the reader'''
        return self.reader.scp_position

    @pos.setter
    def pos(self, pos):
        '''the position setter'''
        self.reader.scp_position = pos




def apply_cmvn(utt, stats):
    '''
    apply mean and variance normalisation

    The mean and variance statistics are computed on previously seen data

    Args:
        utt: the utterance feature numpy matrix
        stats: a numpy array containing the mean and variance statistics. The
            first row contains the sum of all the fautures and as a last element
            the total number of features. The second row contains the squared
            sum of the features and a zero at the end

    Returns:
        a numpy array containing the mean and variance normalized features
    '''

    #compute mean
    mean = stats[0, :-1]/stats[0, -1]

    #compute variance
    variance = stats[1, :-1]/stats[0, -1] - np.square(mean)

    #return mean and variance normalised utterance
    return np.divide(np.subtract(utt, mean), np.sqrt(variance))

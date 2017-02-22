'''@file feature_computer.py
contains the FeatureComputer class'''

from abc import ABCMeta, abstractmethod

class FeatureComputer(object):
    '''A featurecomputer is used to compute features'''

    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''
        FeatureComputer constructor

        Args:
            conf: the feature configuration
        '''

        self.conf = conf

    def __call__(self, sig, rate):
        '''
        compute the features

        Args:


        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

        #compute the features and energy
        feat = self.comp_feat(sig, rate)

        return feat

    @abstractmethod
    def comp_feat(self, sig, rate):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

    @abstractmethod
    def get_dim(self):
        '''the feature dimemsion'''

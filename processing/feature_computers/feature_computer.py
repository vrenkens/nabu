'''@file feature_computer.py
contains the FeatureComputer class'''

from abc import ABCMeta, abstractmethod
import numpy as np
import base

class FeatureComputer(object):
    '''A featurecomputer is used to compute features'''

    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''
        FeatureComputer constructor

        Args:
            conf: the feature configuration
        '''

        if conf['dynamic'] == 'nodelta':
            self.comp_dyn = lambda x: x
        elif conf['dynamic'] == 'delta':
            self.comp_dyn = base.delta
        elif conf['dynamic'] == 'ddelta':
            self.comp_dyn = base.ddelta
        else:
            raise Exception('unknown dynamic type')

        self.conf = conf

    def __call__(self, sig, rate):
        '''
        compute the features

        Args:
            sig: audio signal
            rate: sampling rate

        Returns:
            the features
        '''

        if self.conf['snip_edges'] == 'True':
            #snip the edges
            sig = snip(sig, rate, float(self.conf['winlen']),
                       float(self.conf['winstep']))

        #compute the features and energy
        feat, energy = self.comp_feat(sig, rate)

        #append the energy if requested
        if self.conf['include_energy'] == 'True' and energy is not None:
            feat = np.append(feat, energy[:, np.newaxis], 1)

        #add the dynamic information
        feat = self.comp_dyn(feat)

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

def snip(sig, rate, winlen, winstep):
    '''
    snip the edges of the utterance to fit the sliding window

    Args:
        sig: audio signal
        rate: sampling rate
        winlen: length of the sliding window [s]
        winstep: stepsize of the sliding window [s]

    Returns:
        the snipped signal
    '''
    # calculate the number of frames in the utterance as number of samples in
    #the utterance / number of samples in the frame
    num_frames = int((len(sig)-winlen*rate)/(winstep*rate))
    # cut of the edges to fit the number of frames
    sig = sig[0:int(num_frames*winstep*rate + winlen*rate)]

    return sig

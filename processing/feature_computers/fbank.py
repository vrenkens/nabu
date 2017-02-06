'''@file fbank.py
contains the fbank feature computer'''

import numpy as np
import base
import feature_computer

class Fbank(feature_computer.FeatureComputer):
    '''the feature computer class to compute fbank features'''

    def comp_feat(self, sig, rate):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

        feat, energy = base.logfbank(sig, rate, self.conf)

        if self.conf['include_energy'] == 'True':
            feat = np.append(feat, energy[:, np.newaxis], 1)

        return feat

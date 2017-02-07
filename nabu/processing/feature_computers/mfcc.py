'''@file mfcc.py
contains the fbank feature computer'''

import numpy as np
import base
import feature_computer

class Mfcc(feature_computer.FeatureComputer):
    '''the feature computer class to compute MFCC features'''

    def comp_feat(self, sig, rate):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

        feat, energy = base.mfcc(sig, rate, self.conf)

        if self.conf['include_energy'] == 'True':
            feat = np.append(feat, energy[:, np.newaxis], 1)

        if self.conf['dynamic'] == 'delta':
            feat = base.delta(feat)
        elif self.conf['dynamic'] == 'ddelta':
            feat = base.ddelta(feat)
        elif self.conf['dynamic'] != 'nodelta':
            raise Exception('unknown dynamic type')

        return feat

    def get_dim(self):
        '''the feature dimemsion'''

        dim = int(self.conf['numcep'])

        if self.conf['include_energy'] == 'True':
            dim += 1

        if self.conf['dynamic'] == 'delta':
            dim *= 2
        elif self.conf['dynamic'] == 'ddelta':
            dim *= 3

        return dim

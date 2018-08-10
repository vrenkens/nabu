'''@file audio_processor.py
contains the AudioProcessor class'''


import os
import subprocess
import StringIO
import scipy.io.wavfile as wav
import numpy as np
import processor
from feature_computers import feature_computer_factory

class AudioProcessor(processor.Processor):
    '''a processor for audio files, this will compute features'''

    def __init__(self, conf):
        '''AudioProcessor constructor

        Args:
            conf: processor configuration as a configparser
        '''

        #create the feature computer
        self.comp = feature_computer_factory.factory(
            conf.get('feature', 'feature'))(conf)

        #initialize the metadata
        self.dim = self.comp.get_dim()
        self.max_length = 0
        self.sequence_length_histogram = np.zeros(0, dtype=np.int32)

        super(AudioProcessor, self).__init__(conf)

    def __call__(self, dataline):
        '''process the data in dataline
        Args:
            dataline: either a path to a wav file or a command to read and pipe
                an audio file

        Returns:
            The features as a numpy array'''

        #read the wav file
        rate, utt = _read_wav(dataline)

        #compute the features
        features = self.comp(utt, rate)

        #mean and variance normalize the features
        if self.conf['mvn'] == 'True':
            features = (features-np.mean(features, 0))/np.std(features, 0)

        if self.conf['max_length'] != 'None':
            max_length = int(self.conf['max_length'])
        else:
            max_length = None

        if not max_length or features.shape[0] <= max_length:

            #update the metadata
            self.max_length = max(self.max_length, features.shape[0])
            seq_length = features.shape[0]
            if seq_length >= self.sequence_length_histogram.shape[0]:
                self.sequence_length_histogram = np.concatenate(
                    [self.sequence_length_histogram, np.zeros(
                        seq_length-self.sequence_length_histogram.shape[0]+1,
                        dtype=np.int32)]
                )
            self.sequence_length_histogram[seq_length] += 1

            return features

        else:
            return None

    def write_metadata(self, datadir):
        '''write the processor metadata to disk

        Args:
            dir: the directory where the metadata should be written'''

        with open(os.path.join(datadir, 'sequence_length_histogram.npy'),
                  'w') as fid:
            np.save(fid, self.sequence_length_histogram)
        with open(os.path.join(datadir, 'max_length'), 'w') as fid:
            fid.write(str(self.max_length))
        with open(os.path.join(datadir, 'dim'), 'w') as fid:
            fid.write(str(self.dim))

def _read_wav(wavfile):
    '''
    read a wav file

    Args:
        wavfile: either a path to a wav file or a command to read and pipe
            an audio file

    Returns:
        - the sampling rate
        - the utterance as a numpy array
    '''

    if os.path.exists(wavfile):
        #its a file
        (rate, utterance) = wav.read(wavfile)
    elif wavfile[-1] == '|':
        #its a command

        #read the audio file
        pid = subprocess.Popen(wavfile + ' tee', shell=True,
                               stdout=subprocess.PIPE)
        output, _ = pid.communicate()
        output_buffer = StringIO.StringIO(output)
        (rate, utterance) = wav.read(output_buffer)
    else:
        #its a segment of an utterance
        split = wavfile.split(' ')
        begin = float(split[-2])
        end = float(split[-1])
        unsegmented = ' '.join(split[:-2])
        rate, full_utterance = _read_wav(unsegmented)
        utterance = full_utterance[int(begin*rate):int(end*rate)]


    return rate, utterance

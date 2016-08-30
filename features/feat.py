import sys
sys.path.append('features/python_speech_features')
import features
import scipy.io.wavfile as wav
import numpy as np

class FeatureComputer(object):
	#create the FeatureComputer object
	#	featureType	type of features, current options are mfcc, fbank and ssc
	#	conf: feature configuration
	def __init__(self, featureType, conf):
		if featureType == 'fbank':
			self.compFeat = features.logfbank
		elif featureType == 'mfcc':
			self.compFeat = features.mfcc
		elif featureType == 'ssc':
			self.compFeat = features.ssc
		else:
			raise Exception('unknown feature type')
			
		self.conf = conf
			
	#compute the features
	#	sig: audio signal
	#	rate: sampling rate
	#	returns: the features
	def __call__(self, sig, rate):
		
		if self.conf['snip_edges'] == 'True':
			#snip the edges
			sig = snip(sig, rate, float(self.conf['winlen']), float(self.conf['winstep']))
	
		#compute the features and energy
		feat, energy = self.compFeat(sig, rate, self.conf)
		
		#append the energy if requested
		if self.conf['include_energy'] == 'True':
			feat = np.append(feat,energy[:,np.newaxis],1)
			
		return feat

		

#snip the edges of the utterance to fit the sliding window
#	sig: audio signal
#	rate: sampling rate
#	winlen: length of the sliding window [s]
#	winstep: stepsize of the sliding window [s]
def snip(sig, rate, winlen, winstep):
	# calculate the number of frames in the utterance as number of samples in the utterance / number of samples in the frame
	num_frames = int((len(sig)-winlen*rate)/(winstep*rate)) 
	# cut of the edges to fit the number of frames
	sig = sig[0:int(num_frames*winstep*rate + winlen*rate)]
	
	return sig

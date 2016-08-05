import sys
sys.path.append('python_speech_features')
from features import logfbank
from features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

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

#Compute fbank features
#	sig: audio signal
#	rate: sampling rate
#	winlen: length of the sliding window [s]
#	winstep: stepsize of the sliding window [s]
#	nfilt: number of mel filters
#	nfft: number of fft bins
#	lowfreq: low cuttof frequency
# highfreq: high cuttoff
# preemph: pre-emphesis coefficient
# include_energy: if set to True the energy will be appended to the features
# snip_edges: if set to true the signal will be cut so the length is appropriate for the sliding window length and step, otherwise the window will go over the edge and pad the signal
def compute_fbank(sig, rate, winlen=0.025,winstep=0.01, nfilt=39, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, include_energy=True, snip_edges = True):
	
	if snip_edges:
		#snip the edges
		sig = snip(sig, rate, winlen, winstep)
	
	#compute fbank features and energy
	(feat,energy) = logfbank(sig, rate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
	
	if include_energy:
		#append the energy
		fbank_feat = np.ndarray(shape=(feat.shape[0], feat.shape[1] + 1))
		fbank_feat[:,0:feat.shape[1]] = feat
		fbank_feat[:,feat.shape[1]] = energy
	else:
		fbank_feat = feat		
	
	return fbank_feat

#Compute mfcc features
#	sig: audio signal
#	rate: sampling rate
#	winlen: length of the sliding window [s]
#	winstep: stepsize of the sliding window [s]
# numcep: number of cepstral bins
#	nfilt: number of mel filters
#	nfft: number of fft bins
#	lowfreq: low cuttof frequency
# highfreq: high cuttoff
# preemph: pre-emphesis coefficient
#	ceplifter: alue of the cepstral lifter
# include_energy: if set to True the energy will be appended to the features
# snip_edges: if set to true the signal will be cut so the length is appropriate for the sliding window length and step, otherwise the window will go over the edge and pad the signal
def compute_mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep = 12, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, include_energy=True, snip_edges = True):
	
	if snip_edges:
		#snip the edges
		sig = snip(sig, rate, winlen, winstep)
	
	return mfcc(sig, rate, winlen, winstep, numcep, nfilt, nfft, lowfreq, highfreq, preemph, ceplifter, include_energy)

import numpy as np
import gzip
import feat
import kaldi_io
import scipy.io.wavfile as wav
import cPickle as pickle
import os
from shutil import copyfile
import sys
from random import shuffle


#This function will compute the features of all segments and save them on disk
#	datadir: directory where the kaldi data prep has been done
#	featdir: directory where the features will be put
#	conf: feature configuration
#	feat_type: type of features to be computed
def prepare_data(datadir, featdir, conf, feat_type):
	
	if not os.path.exists(featdir):
		os.makedirs(featdir)
	
	#read the segments
	if os.path.isfile(datadir + '/segments'):
		segments = kaldi_io.read_segments(datadir + '/segments')
		found_segments = True
	else:
		print('WARNING: no segments file found, assuming each wav file is seperate utterance')
		found_segments = False

	#read the wavfiles
	wavfiles = kaldi_io.read_wavfiles(datadir + '/wav.scp')
	
	#create kaldi ark writer
	writer = kaldi_io.KaldiWriteOut(featdir + '/feats.scp')
	if os.path.isfile(featdir + '/feats.ark'):
		os.remove(featdir + '/feats.ark')
		
	if conf['highfreq'] == '-1':
		highfreq = None
	else:
		highfreq = float(conf['highfreq'])
	
	#compute all features and write in ark format
	if found_segments:
		for utt in segments:
			if utt not in wavfiles:
				print('WARNING: ' + utt + ' was not found in the wav files, skipping')
				continue
			if wavfiles[utt][1]:
				#read the audio file and temporarily copy it to tmp (and duplicate, I don't know how to avoid this)
				os.system(wavfiles[utt][0] + ('tee %s/tmp.wav > %s/duplicate.wav'%(featdir, featdir)))
				#read the created wav file
				(rate,utterance) = wav.read(featdir + '/tmp.wav')
				#delete the create file
				os.remove(featdir + '/tmp.wav')
				os.remove(featdir + '/duplicate.wav')
			else:
				(rate,utterance) = wav.read(wavfiles[utt][0])
			for seg in segments[utt]:
				if feat_type == 'fbank':
					features = feat.compute_fbank(utterance[int(seg[1]*rate):int(seg[2]*rate)], rate, float(conf['winlen']), float(conf['winstep']), int(conf['nfilt_fbank']), int(conf['nfft']), float(conf['lowfreq']), highfreq, float(conf['preemph']), conf['include_energy'] == 'True', conf['snip_edges'] == 'True')
				elif feat_type == 'mfcc':
					features = feat.compute_mfcc(utterance[int(seg[1]*rate):int(seg[2]*rate)], rate, float(conf['winlen']), float(conf['winstep']), int(conf['numcep']), int(conf['nfilt']), int(conf['nfft']), float(conf['lowfreq']), highfreq, float(conf['preemph']), float(conf['ceplifter']), conf['include_energy'] == 'True', conf['snip_edges'] == 'True')
				else:
					raise Exception('unknown feature type')
				writer.write_next_utt(featdir + '/feats.ark', seg[0], features)
	else:
		for utt in wavfiles:
			if wavfiles[utt][1]:
				#read the audio file and temporarily copy it to tmp (and duplicate, I don't know how to avoid this)
				os.system(wavfiles[utt][0] + ('tee %s/tmp.wav > %s/duplicate.wav'%(featdir, featdir)))
				#read the created wav file
				(rate,utterance) = wav.read(featdir + '/tmp.wav')
				#delete the create file
				os.remove(featdir + '/tmp.wav')
				os.remove(featdir + '/duplicate.wav')
			else:
				(rate,utterance) = wav.read(wavfiles[utt][0])
			if feat_type == 'fbank':
				features = feat.compute_fbank(utterance, rate, float(conf['winlen']), float(conf['winstep']), int(conf['nfilt_fbank']), int(conf['nfft']), float(conf['lowfreq']), highfreq, float(conf['preemph']), conf['include_energy'] == 'True', conf['snip_edges'] == 'True')
			elif feat_type == 'mfcc':
				features = feat.compute_mfcc(utterance, rate, float(conf['winlen']), float(conf['winstep']), int(conf['numcep']), int(conf['nfilt']), int(conf['nfft']), float(conf['lowfreq']), highfreq, float(conf['preemph']), float(conf['ceplifter']), conf['include_energy'] == 'True', conf['snip_edges'] == 'True')
			else:
				raise Exception('unknown feature type')
			
			writer.write_next_utt(featdir + '/feats.ark', utt, features)

	writer.close()
	
	#copy some kaldi files to features dir
	copyfile(datadir + '/utt2spk', featdir + '/utt2spk')
	copyfile(datadir + '/spk2utt', featdir + '/spk2utt')
	copyfile(datadir + '/text', featdir + '/text')
	copyfile(datadir + '/wav.scp', featdir + '/wav.scp')
	
def compute_cmvn(featdir):
	#read the spk2utt file
	spk2utt = open(featdir + '/spk2utt', 'r')
	
	#create feature reader
	reader = kaldi_io.KaldiReadIn(featdir + '/feats.scp')
	
	#create writer for cmvn stats
	writer = kaldi_io.KaldiWriteOut(featdir + '/cmvn.scp')
	
	#loop over speakers
	for line in spk2utt:
		#cut off end of line character
		line = line[0:len(line)-1] 
	
		split = line.split(' ')
		
		#get first speaker utterance
		spk_data = reader.read_utt(split[1])
		
		#get the rest of the utterances
		for utt_id in split[2:len(split)]:
			spk_data = np.append(spk_data, reader.read_utt(utt_id), axis=0)
			
		#compute mean and variance
		stats = np.zeros([2,spk_data.shape[1]+1])
		stats[0,0:spk_data.shape[1]] = np.sum(spk_data, 0)
		stats[1,0:spk_data.shape[1]] = np.sum(np.square(spk_data),0)
		stats[0, spk_data.shape[1]] = spk_data.shape[0]
		
		#write stats to file
		writer.write_next_utt(featdir + '/cmvn.ark', split[0], stats)
	
#this function will shuffle the utterances
#	featdir: directory where the features can be found
#	valid_size: size of the validation set if it is chosen as part of the training set
def shuffle_examples(featdir,valid_size=0):
	#read feats.scp
	featsfile = open(featdir + '/feats.scp', 'r')
	feats = featsfile.readlines()
	
	#shuffle feats randomly
	shuffle(feats)
	
	#create the validation set
	valid_file = open(featdir + '/feats_validation.scp', 'w')
	valid_file.writelines(feats[0:valid_size])
	
	#wite them to feats_shuffled.scp
	feats_shuffledfile = open(featdir + '/feats_shuffled.scp', 'w')
	feats_shuffledfile.writelines(feats[valid_size:len(feats)])
	
	
		
	
		
	
	


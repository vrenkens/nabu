from six.moves import configparser
import prepare_data
import os
import nnet
import kaldi_io
import cPickle as pickle
import gzip

#here you can set which steps should be executed. If a step has been executed in the past the result have been saved and the step does not have to be executed again (if nothing has changed)
GMMTRAINFEATURES = False 	#required 
GMMTESTFEATURES = False 	#required if the performance of the GMM is tested
DNNTRAINFEATURES = False 	#required
DNNTESTFEATURES = False 	#required if the performance of the DNN is tested
TRAIN_MONO = False 			#required
ALIGN_MONO = False			#required if the monophone GMM is used for alignments
TEST_MONO = False 			#required if the performance of the monphone GMM is tested
TRAIN_TRI = False			#required if the triphone or LDA GMM is used for alignments
ALIGN_TRI = False			#required if the triphone GMM is used for alignments
TEST_TRI = False			#required if the performance of the triphone GMM is tested
TRAIN_LDA = False			#required if the LDA GMM is used for alignments
ALIGN_LDA = False			#required if the LDA GMM is used for alignments
TEST_LDA = False			#required if the performance of the LDA GMM is tested
TRAIN_NNET = True			#required
TEST_NNET = False			#required if the performance of the DNN is tested

#read config file
config = configparser.ConfigParser()
config.read('config/config_AURORA4.cfg')
current_dir = os.getcwd()

#compute the features of the training set for GMM training
if GMMTRAINFEATURES:
	feat_cfg = dict(config.items('gmm-features'))

	print('------- computing GMM training features ----------')
	prepare_data.prepare_data(config.get('directories','train_data'), config.get('directories','train_features') + '/' + feat_cfg['name'],feat_cfg, feat_cfg['type'])
	
	print('------- computing cmvn stats ----------')
	if config.get('gmm-features','apply_cmvn'):
		#compute cmvn stats
		prepare_data.compute_cmvn(config.get('directories','train_features') + '/' + feat_cfg['name'])	
	else:
		#create fake cmvn statistics
		#change directory to kaldi egs
		os.chdir(config.get('directories','kaldi_egs'))
		#create fake cmvn stats
		os.system('steps/compute_cmvn_stats.sh --fake %s %s/cmvn %s' % (config.get('directories','train_features') + '/' + feat_cfg['name'], config.get('directories','expdir'), config.get('directories','train_features') + '/' + feat_cfg['name']))
		#go back to working dir
		os.chdir(current_dir)

#compute the features of the training set for DNN training if they are different then the GMM features
if DNNTRAINFEATURES:		
	if config.get('dnn-features','name') != config.get('gmm-features','name'):
		feat_cfg = dict(config.items('dnn-features'))
		print('------- computing DNN training features ----------')
		prepare_data.prepare_data(config.get('directories','train_data'), config.get('directories','train_features') + '/' + feat_cfg['name'],feat_cfg, feat_cfg['type'])
	
		print('------- computing cmvn stats ----------')
		if config.get('dnn-features','apply_cmvn'):
			#compute cmvn stats
			prepare_data.compute_cmvn(config.get('directories','train_features') + '/' + feat_cfg['name'])	
		else:
			#create fake cmvn statistics
			#change directory to kaldi egs
			os.chdir(config.get('directories','kaldi_egs'))
			#create fake cmvn stats
			os.system('steps/compute_cmvn_stats.sh --fake %s %s/cmvn %s' % (config.get('directories','train_features') + '/' + feat_cfg['name'], config.get('directories','expdir'), config.get('directories','train_features') + '/' + feat_cfg['name']))
			#go back to working dir
			os.chdir(current_dir)
	
	
#compute the features of the training set for GMM testing
if GMMTESTFEATURES:
	feat_cfg = dict(config.items('gmm-features'))

	print('------- computing GMM testing features ----------')
	prepare_data.prepare_data(config.get('directories','test_data'), config.get('directories','test_features') + '/' + feat_cfg['name'],feat_cfg, feat_cfg['type'])
	
	print('------- computing cmvn stats ----------')
	if config.get('gmm-features','apply_cmvn'):
		#compute cmvn stats
		prepare_data.compute_cmvn(config.get('directories','test_features') + '/' + feat_cfg['name'])	
	else:
		#create fake cmvn statistics
		#change directory to kaldi egs
		os.chdir(config.get('directories','kaldi_egs'))
		#create fake cmvn stats
		os.system('steps/compute_cmvn_stats.sh --fake %s %s/cmvn %s' % (config.get('directories','test_features') + '/' + feat_cfg['name'], config.get('directories','expdir'), config.get('directories','test_features') + '/' + feat_cfg['name']))
		#go back to working dir
		os.chdir(current_dir)
	
#compute the features of the training set for DNN testing if they are different then the GMM features
if DNNTESTFEATURES:		
	if config.get('dnn-features','name') != config.get('gmm-features','name'):
		feat_cfg = dict(config.items('dnn-features'))
		print('------- computing DNN testing features ----------')
		prepare_data.prepare_data(config.get('directories','test_data'), config.get('directories','test_features') + '/' + feat_cfg['name'],feat_cfg, feat_cfg['type'])
	
		print('------- computing cmvn stats ----------')
		if config.get('dnn-features','apply_cmvn'):
			#compute cmvn stats
			prepare_data.compute_cmvn(config.get('directories','test_features') + '/' + feat_cfg['name'])	
		else:
			#create fake cmvn statistics
			#change directory to kaldi egs
			os.chdir(config.get('directories','kaldi_egs'))
			#create fake cmvn stats
			os.system('steps/compute_cmvn_stats.sh --fake %s %s/cmvn %s' % (config.get('directories','test_features') + '/' + feat_cfg['name'], config.get('directories','expdir'), config.get('directories','test_features') + '/' + feat_cfg['name']))
			#go back to working dir
			os.chdir(current_dir)
	

#use kaldi to train the monophone GMM
if TRAIN_MONO:
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))

	#train monophone GMM
	print('------- training monophone GMM ----------')
	os.system('steps/train_mono.sh --nj %s --cmd %s --config %s/config/mono.conf %s %s %s/%s' % (config.get('general','num_jobs'), config.get('general','cmd'), current_dir, config.get('directories','train_features') + '/' + config.get('gmm-features','name'), config.get('directories','language'), config.get('directories','expdir'), config.get('mono_gmm','name')))
	
	#build decoding graphs
	print('------- building decoding graphs ----------')
	os.system('utils/mkgraph.sh --mono %s %s/%s %s/%s/graph' % (config.get('directories','language_test'), config.get('directories','expdir'), config.get('mono_gmm','name'), config.get('directories','expdir'), config.get('mono_gmm','name')))
	
	#go back to working dir
	os.chdir(current_dir)


if ALIGN_MONO:
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))

	#align the data
	print('------- aligning the data ----------')
	os.system('steps/align_si.sh --nj %s --cmd %s --config %s/config/ali_mono.conf %s %s %s/%s %s/%s/ali' % (config.get('general','num_jobs'), config.get('general','cmd'), current_dir, config.get('directories','train_features') + '/' + config.get('gmm-features','name'), config.get('directories','language'), config.get('directories','expdir'), config.get('mono_gmm','name'), config.get('directories','expdir'), config.get('mono_gmm','name')))
	
	#convert alignments (transition-ids) to pdf-ids
	for i in range(int(config.get('general','num_jobs'))):
		print('gunzip -c %s/%s/ali/ali.%d.gz | ali-to-pdf %s/%s/ali/final.mdl ark:- ark,t:- | gzip >  %s/%s/ali/pdf.%d.gz' % (config.get('directories','expdir'), config.get('mono_gmm','name'), i+1, config.get('directories','expdir'), config.get('mono_gmm','name'), config.get('directories','expdir'), config.get('mono_gmm','name'), i+1))
		os.system('gunzip -c %s/%s/ali/ali.%d.gz | ali-to-pdf %s/%s/ali/final.mdl ark:- ark,t:- | gzip >  %s/%s/ali/pdf.%d.gz' % (config.get('directories','expdir'), config.get('mono_gmm','name'), i+1, config.get('directories','expdir'), config.get('mono_gmm','name'), config.get('directories','expdir'), config.get('mono_gmm','name'), i+1))
	
	#go back to working dir
	os.chdir(current_dir)


#use kaldi to test the monophone gmm
if TEST_MONO:
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))
	
	#decode using kaldi
	print('------- testing monophone GMM ----------')
	os.system('steps/decode.sh --cmd %s --nj %s %s/%s/graph %s %s/%s/decode | tee %s/%s/decode.log || exit 1;' % (config.get('general','cmd'), config.get('general','num_jobs'), config.get('directories','expdir'), config.get('mono_gmm','name'), config.get('directories','test_features') + '/' + config.get('gmm-features','name'), config.get('directories','expdir'), config.get('mono_gmm','name'), config.get('directories','expdir'), config.get('mono_gmm','name')))
	
	#get results
	os.system('grep WER %s/%s/decode/wer_* | utils/best_wer.sh' % (config.get('directories','expdir'), config.get('mono_gmm','name')))
	
	#go back to working dir
	os.chdir(current_dir)


#use kaldi to train the triphone GMM
if TRAIN_TRI:
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))
	
	#train triphone GMM
	print('------- training triphone GMM ----------')
	os.system('steps/train_deltas.sh --cmd %s --config %s/config/tri.conf %s %s %s %s %s/%s/ali %s/%s' % (config.get('general','cmd'), current_dir, config.get('tri_gmm','num_leaves'), config.get('tri_gmm','tot_gauss'), config.get('directories','train_features') + '/' + config.get('gmm-features','name'), config.get('directories','language'), config.get('directories','expdir'), config.get('mono_gmm','name'), config.get('directories','expdir'), config.get('tri_gmm','name')))
	
	#build decoding graphs
	print('------- building decoding graphs ----------')
	os.system('utils/mkgraph.sh %s %s/%s %s/%s/graph' % (config.get('directories','language_test'), config.get('directories','expdir'), config.get('tri_gmm','name'), config.get('directories','expdir'), config.get('tri_gmm','name')))
	
	#go back to working dir
	os.chdir(current_dir)

if ALIGN_TRI:
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))
		
	#align the data
	print('------- aligning the data ----------')
	os.system('steps/align_si.sh --nj %s --cmd %s --config %s/config/ali_tri.conf %s %s %s/%s %s/%s/ali' % (config.get('general','num_jobs'), config.get('general','cmd'), current_dir, config.get('directories','train_features') + '/' + config.get('gmm-features','name'), config.get('directories','language'), config.get('directories','expdir'), config.get('tri_gmm','name'), config.get('directories','expdir'), config.get('tri_gmm','name')))
	
	#convert alignments (transition-ids) to pdf-ids
	for i in range(int(config.get('general','num_jobs'))):
		os.system('gunzip -c %s/%s/ali/ali.%d.gz | ali-to-pdf %s/%s/ali/final.mdl ark:- ark,t:- | gzip >  %s/%s/ali/pdf.%d.gz' % (config.get('directories','expdir'), config.get('tri_gmm','name'), i+1, config.get('directories','expdir'), config.get('tri_gmm','name'), config.get('directories','expdir'), config.get('tri_gmm','name'), i+1))
	
	#go back to working dir
	os.chdir(current_dir)


#use kaldi to test the triphone GMM
if TEST_TRI:
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))
	
	#decode using kaldi
	print('------- testing triphone GMM ----------')
	os.system('steps/decode.sh --cmd %s --nj %s %s/%s/graph %s %s/%s/decode | tee %s/%s/decode.log || exit 1;' % (config.get('general','cmd'), config.get('general','num_jobs'), config.get('directories','expdir'), config.get('tri_gmm','name'), config.get('directories','test_features') + '/' + config.get('gmm-features','name'), config.get('directories','expdir'), config.get('tri_gmm','name'), config.get('directories','expdir'), config.get('tri_gmm','name')))
	
	#get results
	os.system('grep WER %s/%s/decode/wer_* | utils/best_wer.sh' % (config.get('directories','expdir'), config.get('tri_gmm','name')))
	
	#go back to working dir
	os.chdir(current_dir)
	

#use kaldi to train the LDA+MLLT GMM
if TRAIN_LDA:
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))
	
	#train triphone GMM
	print('------- training LDA+MLLT GMM ----------')
	os.system('steps/train_lda_mllt.sh --cmd %s --config %s/config/lda_mllt.conf --context-opts "--context_width=%s" %s %s %s %s %s/%s/ali %s/%s' % (config.get('general','cmd'), current_dir, config.get('lda_mllt','context_width'), config.get('lda_mllt','num_leaves'), config.get('lda_mllt','tot_gauss'), config.get('directories','train_features') + '/' + config.get('gmm-features','name'), config.get('directories','language'), config.get('directories','expdir'), config.get('tri_gmm','name'), config.get('directories','expdir'), config.get('lda_mllt','name')))
	
	#build decoding graphs
	print('------- building decoding graphs ----------')
	os.system('utils/mkgraph.sh %s %s/%s %s/%s/graph' % (config.get('directories','language_test'), config.get('directories','expdir'), config.get('lda_mllt','name'), config.get('directories','expdir'), config.get('lda_mllt','name')))
	
	#go back to working dir
	os.chdir(current_dir)

if ALIGN_LDA:	
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))

	#align the data
	print('------- aligning the data ----------')
	os.system('steps/align_si.sh --nj %s --cmd %s --config %s/config/ali_lda_mllt.conf %s %s %s/%s %s/%s/ali' % (config.get('general','num_jobs'), config.get('general','cmd'), current_dir, config.get('directories','train_features') + '/' + config.get('gmm-features','name'), config.get('directories','language'), config.get('directories','expdir'), config.get('lda_mllt','name'), config.get('directories','expdir'), config.get('lda_mllt','name')))
	
	#convert alignments (transition-ids) to pdf-ids
	for i in range(int(config.get('general','num_jobs'))):
		os.system('gunzip -c %s/%s/ali/ali.%d.gz | ali-to-pdf %s/%s/ali/final.mdl ark:- ark,t:- | gzip >  %s/%s/ali/pdf.%d.gz' % (config.get('directories','expdir'), config.get('lda_mllt','name'), i+1, config.get('directories','expdir'), config.get('lda_mllt','name'), config.get('directories','expdir'), config.get('lda_mllt','name'), i+1))
	
	#go back to working dir
	os.chdir(current_dir)

#use kaldi to test the LDA+MLLT GMM
if TEST_LDA:
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))
	
	#decode using kaldi
	print('------- testing LDA+MLLT GMM ----------')
	os.system('steps/decode.sh --cmd %s --nj %s %s/%s/graph %s %s/%s/decode | tee %s/%s/decode.log || exit 1;' % (config.get('general','cmd'), config.get('general','num_jobs'), config.get('directories','expdir'), config.get('lda_mllt','name'), config.get('directories','test_features') + '/' + config.get('gmm-features','name'), config.get('directories','expdir'), config.get('lda_mllt','name'), config.get('directories','expdir'), config.get('lda_mllt','name')))
	
	#get results
	os.system('grep WER %s/%s/decode/wer_* | utils/best_wer.sh' % (config.get('directories','expdir'), config.get('lda_mllt','name')))
	
	#go back to working dir
	os.chdir(current_dir)

#get the feature input dim
reader = kaldi_io.KaldiReadIn(config.get('directories','train_features') + '/' + config.get('dnn-features','name') + '/feats.scp')
(_,features,_) = reader.read_next_utt()
input_dim = features.shape[1]
		
#read the utterance to speaker mapping
utt2spk = kaldi_io.read_utt2spk(config.get('directories','train_features') + '/' +  config.get('dnn-features','name') + '/utt2spk')

#get number of output labels
numpdfs = open(config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/graph/num_pdfs')
num_labels = numpdfs.read()
num_labels = int(num_labels[0:len(num_labels)-1])
numpdfs.close()
	
#create the neural net 	
Nnet = nnet.Nnet(config, input_dim, num_labels)

if TRAIN_NNET:

	#only shuffle if we start with initialisation
	if config.get('nnet','starting_step') == '0':
		#shuffle the examples on disk
		print('------- shuffling examples ----------')
		prepare_data.shuffle_examples(config.get('directories','train_features') + '/' +  config.get('dnn-features','name'))
	
	#read the alignments
	print('------- reading alignments ----------')
	alignments = {}
	for i in range(int(config.get('general','num_jobs'))):
		alignments.update(kaldi_io.read_alignments(config.get('directories','expdir') + '/' + config.get('nnet','gmm_name') + '/ali/pdf.' + str(i+1) + '.gz'))
	
	#train the neural net
	print('------- training neural net ----------')
	Nnet.train(config.get('directories','train_features') + '/' +  config.get('dnn-features','name'), alignments, utt2spk)
		

if TEST_NNET:
	#read the utterance to speaker mapping
	print('------- reading utt2spk ----------')
	utt2spk = kaldi_io.read_utt2spk(config.get('directories','test_features') + '/' +  config.get('dnn-features','name') + '/utt2spk')

	#use the neural net to calculate posteriors for the testing set
	print('------- computing state pseudo-likelihoods ----------')
	savedir = config.get('directories','expdir') + '/' + config.get('nnet','name')
	decodedir = savedir + '/decode'
	if not os.path.isdir(decodedir):
		os.mkdir(decodedir)
	Nnet.decode(config.get('directories','test_features') + '/' +  config.get('dnn-features','name'), utt2spk, savedir, decodedir)
	
	#create a dummy neural net
	print('------- creating dummy nnet ----------')
	kaldi_io.create_dummy(config.get('directories','expdir') + '/' + config.get('nnet','gmm_name'), decodedir, config.get('directories','test_features') + '/' +  config.get('dnn-features','name'), nnet_cfg['num_labels'])
	
	#change directory to kaldi egs
	os.chdir(config.get('directories','kaldi_egs'))
	
	#build decoding graphs
	print('------- building decoding graphs ----------')
	if config.get('nnet','monophone') == 'True':
		os.system('utils/mkgraph.sh --mono %s %s %s/graph' % (config.get('directories','language_test'), decodedir, decodedir))
	else:
		os.system('utils/mkgraph.sh %s %s %s/graph' % (config.get('directories','language_test'), decodedir, decodedir))
	
	#create fake cmvn stats
	os.system('steps/compute_cmvn_stats.sh --fake %s %s/cmvn %s' % (decodedir, config.get('directories','expdir'), decodedir))
	
	#decode using kaldi
	print('------- decoding testing sets ----------')
	os.system('steps/nnet2/decode.sh --cmd %s --nj %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % (config.get('general','cmd'), config.get('general','num_jobs'), decodedir, decodedir, decodedir, decodedir))
	
	#get results
	os.system('grep WER %s/kaldi_decode/wer_* | utils/best_wer.sh' % decodedir)
	
	#go back to working dir
	os.chdir(current_dir)
	
	
	
	
	


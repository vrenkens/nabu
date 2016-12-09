'''@file main.py
run this file to go through the neural net training procedure, look at the config files in the config directory to modify the settings'''

import os
from six.moves import configparser
from neuralnetworks import nnet
from processing import ark, prepare_data, feature_reader, batchdispenser, target_coder, target_normalizers, score

import pdb

#here you can set which steps should be executed. If a step has been executed in the past the result have been saved and the step does not have to be executed again (if nothing has changed)
TRAINFEATURES = False
DEVFEATURES = False
TESTFEATURES = False
TRAIN = True
TEST = True

#pointers to the config files
database_cfg_file = 'config/databases/TIMIT.cfg'
feat_cfg_file = 'config/features/fbank.cfg'
nnet_cfg_file = 'config/nnet/DBLSTM_CTC.cfg'

#set the CUDA GPU that Tensorflow should use
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#read the database config file
database_cfg = configparser.ConfigParser()
database_cfg.read(database_cfg_file)
database_cfg = dict(database_cfg.items('directories'))

#read the features config file
feat_cfg = configparser.ConfigParser()
feat_cfg.read(feat_cfg_file)
feat_cfg = dict(feat_cfg.items('features'))

#compute the features of the training set for training
if TRAINFEATURES:
    print '------- computing training features ----------'
    prepare_data.prepare_data(database_cfg['train_data'], database_cfg['train_features'] + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(database_cfg['train_features'] + '/' + feat_cfg['name'])

#compute the features of the dev set
if DEVFEATURES:
    if 'dev_data' in database_cfg:

        print '------- computing developement features ----------'
        prepare_data.prepare_data(database_cfg['dev_data'], database_cfg['dev_features'] + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

        print '------- computing cmvn stats ----------'
        prepare_data.compute_cmvn(database_cfg['dev_features'] + '/' + feat_cfg['name'])

#compute the features of the test set for testing
if TESTFEATURES:
    print '------- computing testing features ----------'
    prepare_data.prepare_data(database_cfg['test_data'], database_cfg['test_features'] + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(database_cfg['test_features'] + '/' + feat_cfg['name'])

#get the feature input dim
reader = ark.ArkReader(database_cfg['train_features'] + '/' + feat_cfg['name'] + '/feats.scp')
_, features, _ = reader.read_next_utt()
input_dim = features.shape[1]

#create the coder
coder = target_coder.TextCoder(target_normalizers.aurora4_normalizer)

#create the neural net
nnet_cfg = configparser.ConfigParser()
nnet_cfg.read(nnet_cfg_file)
nnet_cfg = dict(nnet_cfg.items('nnet'))
savedir = database_cfg['expdir'] + '/' + nnet_cfg['name']
nnet = nnet.Nnet(nnet_cfg, savedir, input_dim, coder.num_labels)

if TRAIN:

    featdir = database_cfg['train_features'] + '/' +  feat_cfg['name']

    #only shuffle if we start with initialisation
    if nnet_cfg['starting_step'] == '0':
        #shuffle the examples on disk
        print '------- shuffling examples ----------'
        prepare_data.shuffle_examples(featdir)


    #create a feature reader for the training data
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(featdir + '/feats_shuffled.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', 0, max_input_length)

    #the path to the text file
    textfile = database_cfg['traintext']

    #create a batch dispenser for the training data
    dispenser = batchdispenser.TextBatchDispenser(featreader, coder, int(nnet_cfg['batch_size']), textfile)

    if 'dev_data' in database_cfg:
        featdir = database_cfg['dev_features'] + '/' +  feat_cfg['name']
        with open(featdir + '/maxlength', 'r') as fid:
            max_input_length = int(fid.read())
        val_featreader = feature_reader.FeatureReader(featdir + '/feats.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', 0, max_input_length)
        val_textfile = database_cfg['devtext']

        val_dispenser = batchdispenser.TextBatchDispenser(val_featreader, coder, int(nnet_cfg['batch_size']), val_textfile)

    else:
        if int(nnet_cfg['valid_utt']) > 0:
            val_dispenser = dispenser.split(int(nnet_cfg['valid_utt']))
        else:
            val_dispenser = None

    #train the neural net
    print '------- training neural net ----------'
    nnet.train(dispenser, val_dispenser)


if TEST:

    #use the neural net to calculate posteriors for the testing set
    print '------- decoding test set ----------'
    decodedir = savedir + '/decode'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    featdir = database_cfg['test_features'] + '/' +  feat_cfg['name']

    #create a feature reader
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(featdir + '/feats.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', 0, max_input_length)

    #decode with te neural net
    resultsfolder = savedir + '/decode'
    nbests = nnet.decode(featreader, coder)

    #the path to the text file
    textfile = database_cfg['testtext']

    #read all the reference transcriptions
    with open(textfile) as fid:
        lines = fid.readlines()

    references = dict()
    for line in lines:
        splitline = line.strip().split('')
        references[splitline[0]] = coder.normalize(splitline[1:])

    #compute the character error rate
    pdb.set_trace()
    CER = score.CER(nbests, references)

    print 'character error rate: %f' % CER

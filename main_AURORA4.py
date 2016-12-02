'''@file main.py
run this file to go through the neural net training procedure, look at the config files in the config directory to modify the settings'''

import os
from six.moves import configparser
from neuralnetworks import nnet
from processing import ark, prepare_data, feature_reader, batchdispenser, target_coder, target_normalizers, score

#here you can set which steps should be executed. If a step has been executed in the past the result have been saved and the step does not have to be executed again (if nothing has changed)
TRAINFEATURES = False
DEVFEATURES = False
TESTFEATURES = False
TRAIN = True
TEST = True

#read config file
config = configparser.ConfigParser()
config.read('config/config_TIMIT.cfg')
current_dir = os.getcwd()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#compute the features of the training set for training
if TRAINFEATURES:
    feat_cfg = dict(config.items('features'))

    print '------- computing training features ----------'
    prepare_data.prepare_data(config.get('directories', 'train_data'), config.get('directories', 'train_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(config.get('directories', 'train_features') + '/' + feat_cfg['name'])

#compute the features of the dev set
if DEVFEATURES:
    if 'dev_data' in dict(config.items('directories')):
        feat_cfg = dict(config.items('features'))

        print '------- computing developement features ----------'
        prepare_data.prepare_data(config.get('directories', 'dev_data'), config.get('directories', 'dev_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

        print '------- computing cmvn stats ----------'
        prepare_data.compute_cmvn(config.get('directories', 'dev_features') + '/' + feat_cfg['name'])

#compute the features of the test set for testing
if TESTFEATURES:
    feat_cfg = dict(config.items('features'))

    print '------- computing testing features ----------'
    prepare_data.prepare_data(config.get('directories', 'test_data'), config.get('directories', 'test_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(config.get('directories', 'test_features') + '/' + feat_cfg['name'])

#get the feature input dim
reader = ark.ArkReader(config.get('directories', 'train_features') + '/' + config.get('features', 'name') + '/feats.scp')
_, features, _ = reader.read_next_utt()
input_dim = features.shape[1]

#create the coder
coder = target_coder.PhonemeEncoder(target_normalizers.timit_phone_norm)

#create the neural net
nnet = nnet.Nnet(config, input_dim, coder.num_labels)

if TRAIN:

    #only shuffle if we start with initialisation
    if config.get('nnet', 'starting_step') == '0':
        #shuffle the examples on disk
        print '------- shuffling examples ----------'
        prepare_data.shuffle_examples(config.get('directories', 'train_features') + '/' +  config.get('features', 'name'))


    #create a feature reader for the training data
    featdir = config.get('directories', 'train_features') + '/' +  config.get('features', 'name')
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(featdir + '/feats_shuffled.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', 0, max_input_length)

    #the path to the text file
    textfile = config.get('directories', 'train_data') + '/text'

    #create a batch dispenser for the training data
    dispenser = batchdispenser.TextBatchDispenser(featreader, coder, int(config.get('nnet', 'batch_size')), textfile)

    if 'dev_data' in dict(config.items('directories')):
        featdir = config.get('directories', 'dev_features') + '/' +  config.get('features', 'name')
        with open(featdir + '/maxlength', 'r') as fid:
            max_input_length = int(fid.read())
        val_featreader = feature_reader.FeatureReader(featdir + '/feats.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', 0, max_input_length)
        val_textfile = config.get('directories', 'dev_data') + '/text'

        val_dispenser = batchdispenser.TextBatchDispenser(val_featreader, coder, int(config.get('nnet', 'batch_size')), val_textfile)

    else:
        if int(config.get('nnet','valid_batches')) > 0:
            val_dispenser = dispenser.split(int(config.get('nnet','valid_utt')))
        else:
            val_dispenser = None

    #train the neural net
    print '------- training neural net ----------'
    nnet.train(dispenser, val_dispenser)


if TEST:

    #use the neural net to calculate posteriors for the testing set
    print '------- decoding test set ----------'
    savedir = config.get('directories', 'expdir') + '/' + config.get('nnet', 'name')
    decodedir = savedir + '/decode'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    featdir = config.get('directories', 'test_features') + '/' +  config.get('features', 'name')

    #create a feature reader
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(featdir + '/feats.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', 0, max_input_length)

    #decode with te neural net
    resultsfolder = savedir + '/decode'
    nbests = nnet.decode(featreader, coder)

    #the path to the text file
    textfile = config.get('directories', 'test_data') + '/text'

    #read all the reference transcriptions
    with open(textfile) as fid:
        lines = fid.readlines()

    references = dict()
    for line in lines:
        splitline = line.strip().split(' ')
        references[splitline[0]] = target_normalizers.timit_phone_norm(
            ' '.join(splitline[1:]), None)

    #compute the character error rate
    CER = score.CER(nbests, references)

    print 'character error rate: %f' % CER

'''@file dataprep.py
this file will go through the feature computation'''


from six.moves import configparser
from processing import ark, prepare_data

#pointers to the config files
database_cfg_file = 'config/databases/TIMIT.cfg'
feat_cfg_file = 'config/features/fbank.cfg'

#read the database config file
database_cfg = configparser.ConfigParser()
database_cfg.read(database_cfg_file)
database_cfg = dict(database_cfg.items('directories'))

#read the features config file
feat_cfg = configparser.ConfigParser()
feat_cfg.read(feat_cfg_file)
feat_cfg = dict(feat_cfg.items('features'))

#compute the features of the training set for training
print '------- computing training features ----------'
prepare_data.prepare_data(
    datadir=database_cfg['train_data'],
    featdir=database_cfg['train_features'] + '/' + feat_cfg['name'],
    conf=feat_cfg)

print '------- computing cmvn stats ----------'
prepare_data.compute_cmvn(
    featdir=database_cfg['train_features'] + '/' + feat_cfg['name'])

#compute the features of the dev set
if 'dev_data' in database_cfg:

    print '------- computing developement features ----------'
    prepare_data.prepare_data(
        datadir=database_cfg['dev_data'],
        featdir=database_cfg['dev_features'] + '/' + feat_cfg['name'],
        conf=feat_cfg)

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(
        featdir=database_cfg['dev_features'] + '/' + feat_cfg['name'])

#compute the features of the test set for testing
print '------- computing testing features ----------'
prepare_data.prepare_data(
    datadir=database_cfg['test_data'],
    featdir=database_cfg['test_features'] + '/' + feat_cfg['name'],
    conf=feat_cfg)

print '------- computing cmvn stats ----------'
prepare_data.compute_cmvn(
    featdir=database_cfg['test_features'] + '/' + feat_cfg['name'])

#get the feature dim
reader = ark.ArkReader(database_cfg['train_features'] + '/' + feat_cfg['name'] +
                       '/feats.scp')
_, features, _ = reader.read_next_utt()
input_dim = features.shape[1]
fid = open(
    database_cfg['train_features'] + '/' + feat_cfg['name'] + '/dim', 'w')
fid.write(str(input_dim))
fid.close()

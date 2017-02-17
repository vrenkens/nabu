'''@file lm_dataprep.py
this file will do the dataprep for lm training'''

import os
from six.moves import configparser
from nabu.processing.target_normalizers import normalizer_factory

#pointer to the confif file
database_cfg_file = 'config/lm_databases/aurora4.conf'

#read the database config file
database_cfg = configparser.ConfigParser()
database_cfg.read(database_cfg_file)
database_cfg = dict(database_cfg.items('database'))

#create the text normalizer
normalizer = normalizer_factory.factory(database_cfg['normalizer'])

print '------- normalizing training text -----------'
sourcefiles = database_cfg['trainfiles'].split(' ')
if not os.path.isdir(database_cfg['train_dir']):
    os.makedirs(database_cfg['train_dir'])
target_fid = open(os.path.join(database_cfg['train_dir'], 'text'), 'w')
max_num_chars = 0
numlines = 0

#read the textfiles line by line, normalize and write in target file
for sourcefile in sourcefiles:
    with open(sourcefile) as fid:
        for line in fid.readlines():
            normalized = normalizer(line.strip())
            max_num_chars = max(max_num_chars, len(normalized.split(' ')))
            target_fid.write('%s\n' % (normalized))
            numlines += 1

#store the alphabet
with open(os.path.join(database_cfg['train_dir'], 'alphabet'), 'w') as fid:
    fid.write(' '.join(normalizer.alphabet))

#store the maximum number of characters
with open(os.path.join(database_cfg['train_dir'], 'max_num_chars'), 'w') as fid:
    fid.write(str(max_num_chars))

#store the number of lines
with open(os.path.join(database_cfg['train_dir'], 'numlines'), 'w') as fid:
    fid.write(str(numlines))

target_fid.close()

print '------- normalizing testing text -----------'
sourcefiles = database_cfg['testfiles'].split(' ')
if not os.path.isdir(database_cfg['test_dir']):
    os.makedirs(database_cfg['test_dir'])
target_fid = open(os.path.join(database_cfg['test_dir'], 'text'), 'w')
max_num_chars = 0
numlines = 0

#read the textfiles line by line, normalize and write in target file
for sourcefile in sourcefiles:
    with open(sourcefile) as fid:
        for line in fid.readlines():
            normalized = normalizer(line.strip())
            max_num_chars = max(max_num_chars, len(normalized.split(' ')))
            target_fid.write('%s\n' % (normalized))
            numlines += 1

#store the alphabet
with open(os.path.join(database_cfg['test_dir'], 'alphabet'), 'w') as fid:
    fid.write(' '.join(normalizer.alphabet))

#store the maximum number of characters
with open(os.path.join(database_cfg['test_dir'], 'max_num_chars'), 'w') as fid:
    fid.write(str(max_num_chars))

#store the number of lines
with open(os.path.join(database_cfg['test_dir'], 'numlines'), 'w') as fid:
    fid.write(str(numlines))

target_fid.close()

if 'devfiles' in database_cfg:
    print '------- normalizing dev text -----------'
    sourcefiles = database_cfg['devfiles'].split(' ')
    if not os.path.isdir(database_cfg['dev_dir']):
        os.makedirs(database_cfg['dev_dir'])
    target_fid = open(os.path.join(database_cfg['dev_dir'], 'text'), 'w')
    max_num_chars = 0
    numlines = 0

    #read the textfiles line by line, normalize and write in target file
    for sourcefile in sourcefiles:
        with open(sourcefile) as fid:
            for line in fid.readlines():
                normalized = normalizer(line.strip())
                max_num_chars = max(max_num_chars, len(normalized.split(' ')))
                target_fid.write('%s\n' % (normalized))
                numlines += 1

    #store the alphabet
    with open(os.path.join(database_cfg['dev_dir'], 'alphabet'), 'w') as fid:
        fid.write(' '.join(normalizer.alphabet))

    #store the maximum number of characters
    with open(os.path.join(database_cfg['dev_dir'], 'max_num_chars'),
              'w') as fid:
        fid.write(str(max_num_chars))

    #store the number of lines
    with open(os.path.join(database_cfg['dev_dir'], 'numlines'), 'w') as fid:
        fid.write(str(numlines))

    target_fid.close()

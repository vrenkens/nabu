'''@file decode.py
this file will use the model to decode a set of data'''

import sys
import os
sys.path.append(os.getcwd())
from six.moves import configparser
from nabu.neuralnetworks.recognizer import Recognizer

def decode(expdir):
    '''does everything for decoding'''

    #read the database config file
    database_cfg = configparser.ConfigParser()
    database_cfg.read(os.path.join(expdir, 'database.cfg'))

    #read the recognizer config file
    recognizer_cfg = configparser.ConfigParser()
    recognizer_cfg.read(os.path.join(expdir, 'recognizer.cfg'))

    #create the recognizer
    recognizer = Recognizer(
        conf=recognizer_cfg,
        dataconf=database_cfg,
        expdir=expdir)

    #do the recognition
    recognizer.recognize()

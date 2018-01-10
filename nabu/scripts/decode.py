'''@file decode.py
this file will use the model to decode a set of data'''

import sys
import os
import cPickle as pickle
sys.path.append(os.getcwd())
from six.moves import configparser
import tensorflow as tf
from nabu.neuralnetworks.recognizer import Recognizer
from nabu.neuralnetworks.models.model import Model

def decode(expdir, testing=False):
    '''does everything for decoding

    args:
        expdir: the experiments directory
        testing: if true only the graph will be created for debugging purposes
    '''

    #read the database config file
    database_cfg = configparser.ConfigParser()
    database_cfg.read(os.path.join(expdir, 'database.conf'))

    #read the recognizer config file
    recognizer_cfg = configparser.ConfigParser()
    recognizer_cfg.read(os.path.join(expdir, 'recognizer.cfg'))


    if testing:
        model_cfg = configparser.ConfigParser()
        model_cfg.read(os.path.join(expdir, 'model.cfg'))
        trainer_cfg = configparser.ConfigParser()
        trainer_cfg.read(os.path.join(expdir, 'trainer.cfg'))
        model = Model(
            conf=model_cfg,
            trainlabels=int(trainer_cfg.get('trainer', 'trainlabels')),
            constraint=None)
    else:
        #load the model
        with open(os.path.join(expdir, 'model', 'model.pkl'), 'rb') as fid:
            model = pickle.load(fid)

    #create the recognizer
    recognizer = Recognizer(
        model=model,
        conf=recognizer_cfg,
        dataconf=database_cfg,
        expdir=expdir)

    if testing:
        return

    #do the recognition
    recognizer.recognize()

if __name__ == '__main__':

    tf.app.flags.DEFINE_string('expdir', 'expdir',
                               'the exeriments directory that was used for'
                               ' training'
                              )
    FLAGS = tf.app.flags.FLAGS

    decode(FLAGS.expdir, False)

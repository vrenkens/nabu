'''@file test.py
this file will test the performance of a model'''

import sys
import os
sys.path.append(os.getcwd())
import cPickle as pickle
from six.moves import configparser
import tensorflow as tf
from nabu.neuralnetworks.evaluators import evaluator_factory
from nabu.neuralnetworks.components.hooks import LoadAtBegin, SummaryHook
from nabu.neuralnetworks.models.model import Model

def test(expdir, testing=False):
    '''does everything for testing

    args:
        expdir: the experiments directory
        testing: if true only the graph will be created for debugging purposes
    '''

    #read the database config file
    database_cfg = configparser.ConfigParser()
    database_cfg.read(os.path.join(expdir, 'database.conf'))

    if testing:
        model_cfg = configparser.ConfigParser()
        model_cfg.read(os.path.join(expdir, 'model.cfg'))
        trainer_cfg = configparser.ConfigParser()
        trainer_cfg.read(os.path.join(expdir, 'trainer.cfg'))
        model = Model(
            conf=model_cfg,
            trainlabels=int(trainer_cfg.get('trainer', 'trainlabels')))
    else:
        #load the model
        with open(os.path.join(expdir, 'model', 'model.pkl'), 'rb') as fid:
            model = pickle.load(fid)

    #read the evaluator config file
    evaluator_cfg = configparser.ConfigParser()
    evaluator_cfg.read(os.path.join(expdir, 'test_evaluator.cfg'))

    #create the evaluator
    evaltype = evaluator_cfg.get('evaluator', 'evaluator')
    evaluator = evaluator_factory.factory(evaltype)(
        conf=evaluator_cfg,
        dataconf=database_cfg,
        model=model)

    #create the graph
    graph = tf.Graph()

    with graph.as_default():

        #compute the loss
        batch_loss, numbatches = evaluator.evaluate()

        if testing:
            return

        #create a histogram for all trainable parameters
        for param in model.variables:
            tf.summary.histogram(param.name, param)

        #create a hook that will load the model
        load_hook = LoadAtBegin(
            os.path.join(expdir, 'model', 'network.ckpt'),
            model.variables)

        #create a hook for summary writing
        summary_hook = SummaryHook(os.path.join(expdir, 'logdir'))

        #start the session
        with tf.train.SingularMonitoredSession(
            hooks=[load_hook, summary_hook]) as sess:

            loss = 0.0
            for _ in range(numbatches):
                loss += batch_loss.eval(session=sess)
            loss = loss/numbatches

    print 'loss = %f' % loss

    #write the result to disk
    with open(os.path.join(expdir, 'result'), 'w') as fid:
        fid.write(str(loss))

if __name__ == '__main__':

    tf.app.flags.DEFINE_string('expdir', 'expdir',
                               'the exeriments directory that was used for'
                               ' training'
                              )
    FLAGS = tf.app.flags.FLAGS

    test(FLAGS.expdir, False)

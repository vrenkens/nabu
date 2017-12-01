'''@file test.py
this file will test the performance of a model'''

import sys
import os
sys.path.append(os.getcwd())
import cPickle as pickle
from six.moves import configparser
import tensorflow as tf
from nabu.neuralnetworks.evaluators import evaluator_factory
from nabu.neuralnetworks.components.hooks import LoadAtBegin
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
        loss, update_loss, numbatches = evaluator.evaluate()

        if testing:
            return

        #create a histogram for all trainable parameters
        for param in model.variables:
            tf.summary.histogram(param.name, param, ['variables'])



        eval_summary = tf.summary.merge_all('eval_summaries')
        variable_summary = tf.summary.merge_all('variables')

        #create a hook that will load the model
        load_hook = LoadAtBegin(
            os.path.join(expdir, 'model', 'network.ckpt'),
            model.variables)

        #start the session
        with tf.train.SingularMonitoredSession(
            hooks=[load_hook]) as sess:

            summary_writer = tf.summary.FileWriter(
                os.path.join(expdir, 'logdir'))

            summary = variable_summary.eval(session=sess)
            summary_writer.add_summary(summary)

            for i in range(numbatches):
                _, summary = sess.run([update_loss, eval_summary])
                summary_writer.add_summary(summary, i)

            loss = loss.eval(session=sess)

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

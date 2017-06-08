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

def test(expdir):
    '''does everything for testing'''

    #read the database config file
    database_cfg = configparser.ConfigParser()
    database_cfg.read(os.path.join(expdir, 'database.cfg'))

    #load the model
    with open(os.path.join(expdir, 'model', 'model.pkl'), 'rb') as fid:
        model = pickle.load(fid)

    #read the evaluator config file
    evaluator_cfg = configparser.ConfigParser()
    evaluator_cfg.read(os.path.join(expdir, 'evaluator.cfg'))

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

        #create a hook that will load the model
        load_hook = LoadAtBegin(
            os.path.join(expdir, 'model', 'network.ckpt'),
            model)

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

    test(FLAGS.expdir)

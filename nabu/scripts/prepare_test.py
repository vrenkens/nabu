'''@file test.py
this file will run the test script'''

import sys
import os
sys.path.append(os.getcwd())
import shutil
import subprocess
from six.moves import configparser
import tensorflow as tf
from test import test


tf.app.flags.DEFINE_string('expdir', None,
                           'the exeriments directory that was used for training'
                          )
tf.app.flags.DEFINE_string('recipe', None,
                           'The directory containing the recipe'
                          )
tf.app.flags.DEFINE_string('computing', 'standart',
                           'the distributed computing system one of standart or'
                           ' condor'
                          )

FLAGS = tf.app.flags.FLAGS

def main(_):
    '''main'''

    if FLAGS.expdir is None:
        raise Exception('no expdir specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(FLAGS.expdir):
        raise Exception('cannot find expdir %s' % FLAGS.expdir)

    if FLAGS.recipe is None:
        raise Exception('no recipe specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(FLAGS.recipe):
        raise Exception('cannot find recipe %s' % FLAGS.recipe)

    evaluator_cfg_file = os.path.join(FLAGS.recipe, 'test_evaluator.cfg')
    database_cfg_file = os.path.join(FLAGS.recipe, 'database.conf')

    #create the testing dir
    if os.path.isdir(os.path.join(FLAGS.expdir, 'test')):
        shutil.rmtree(os.path.join(FLAGS.expdir, 'test'))
    os.makedirs(os.path.join(FLAGS.expdir, 'test'))

    #copy the config files
    shutil.copyfile(database_cfg_file,
                    os.path.join(FLAGS.expdir, 'test', 'database.cfg'))
    shutil.copyfile(evaluator_cfg_file,
                    os.path.join(FLAGS.expdir, 'test', 'evaluator.cfg'))

    #create a link to the model
    os.symlink(os.path.join(FLAGS.expdir, 'model'),
               os.path.join(FLAGS.expdir, 'test', 'model'))

    if FLAGS.computing == 'condor_non_distributed':

        computing_cfg_file = 'config/computing/condor/non_distributed.cfg'
        parsed_computing_cfg = configparser.ConfigParser()
        parsed_computing_cfg.read(computing_cfg_file)
        computing_cfg = dict(parsed_computing_cfg.items('computing'))

        if not os.path.isdir(os.path.join(FLAGS.expdir, 'test', 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'test', 'outputs'))

        subprocess.call(['condor_submit',
                         'expdir=%s' % os.path.join(FLAGS.expdir, 'test'),
                         'script=nabu/scripts/test.py',
                         'memory=%s' % computing_cfg['minmemory'],
                         'nabu/computing/condor/non_distributed.job'])


    elif FLAGS.computing == 'non_distributed':
        test(expdir=os.path.join(FLAGS.expdir, 'test'))

    else:
        raise Exception('Unknown computing type %s' % FLAGS.computing)

if __name__ == '__main__':
    tf.app.run()

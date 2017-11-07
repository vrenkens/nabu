'''@file test.py
this file will run the test script

usage: run test --expdir=<expdir> --recipe=<recipe> --computing=<computing>'''

import sys
import os
sys.path.append(os.getcwd())
import shutil
import subprocess
from six.moves import configparser
import tensorflow as tf
from test import test

def main(expdir, recipe, computing):
    '''main function

    Args:
        - expdir: the training experiments directory
        - recipe: the directory containing the recipe config files
        - computing: the computing type, one off (condor, standart)'''

    if expdir is None:
        raise Exception('no expdir specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(expdir):
        raise Exception('cannot find expdir %s' % expdir)

    if recipe is None:
        raise Exception('no recipe specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')

    if not os.path.isdir(recipe):
        raise Exception('cannot find recipe %s' % recipe)

    evaluator_cfg_file = os.path.join(recipe, 'test_evaluator.cfg')
    database_cfg_file = os.path.join(recipe, 'database.conf')

    #create the testing dir
    if os.path.isdir(os.path.join(expdir, 'test')):
        shutil.rmtree(os.path.join(expdir, 'test'))
    os.makedirs(os.path.join(expdir, 'test'))

    #copy the config files
    shutil.copyfile(database_cfg_file,
                    os.path.join(expdir, 'test', 'database.cfg'))
    shutil.copyfile(evaluator_cfg_file,
                    os.path.join(expdir, 'test', 'evaluator.cfg'))

    #create a link to the model
    os.symlink(os.path.join(expdir, 'model'),
               os.path.join(expdir, 'test', 'model'))

    if computing == 'condor':

        computing_cfg_file = 'config/computing/condor/non_distributed.cfg'
        parsed_computing_cfg = configparser.ConfigParser()
        parsed_computing_cfg.read(computing_cfg_file)
        computing_cfg = dict(parsed_computing_cfg.items('computing'))

        if not os.path.isdir(os.path.join(expdir, 'test', 'outputs')):
            os.makedirs(os.path.join(expdir, 'test', 'outputs'))

        subprocess.call(['condor_submit',
                         'expdir=%s' % os.path.join(expdir, 'test'),
                         'script=nabu/scripts/test.py',
                         'memory=%s' % computing_cfg['minmemory'],
                         'nabu/computing/condor/non_distributed.job'])


    elif computing == 'standart':
        test(expdir=os.path.join(expdir, 'test'))

    else:
        raise Exception('Unknown computing type %s' % computing)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('expdir', None,
                               'the training experiments directory'
                              )
    tf.app.flags.DEFINE_string('recipe', None,
                               'the directory containing the recipe config '
                               'files'
                              )
    tf.app.flags.DEFINE_string('computing', 'standart',
                               'the computing type, one off (condor, standart)'
                              )

    FLAGS = tf.app.flags.FLAGS

    main(
        expdir=FLAGS.expdir,
        recipe=FLAGS.recipe,
        computing=FLAGS.computing)

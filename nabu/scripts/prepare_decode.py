'''@file decode.py
this file will run the decode script'''

import sys
import os
sys.path.append(os.getcwd())
import shutil
import subprocess
from six.moves import configparser
import tensorflow as tf
from decode import decode


tf.app.flags.DEFINE_string('expdir', None,
                           'the exeriments directory that was used for training'
                          )
tf.app.flags.DEFINE_string('recipe', None,
                           'The directory containing the recipe'
                          )
tf.app.flags.DEFINE_string('computing', 'standard',
                           'the distributed computing system one of standard or'
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

    #create the decoding dir
    if os.path.isdir(os.path.join(FLAGS.expdir, 'decode')):
        shutil.rmtree(os.path.join(FLAGS.expdir, 'decode'))
    os.makedirs(os.path.join(FLAGS.expdir, 'decode'))

    recognizer_cfg_file = os.path.join(FLAGS.recipe, 'recognizer.cfg')
    database_cfg_file = os.path.join(FLAGS.recipe, 'database.conf')

    #copy the config files
    shutil.copyfile(database_cfg_file,
                    os.path.join(FLAGS.expdir, 'decode', 'database.conf'))
    shutil.copyfile(recognizer_cfg_file,
                    os.path.join(FLAGS.expdir, 'decode', 'recognizer.cfg'))

    #create a link to the model
    os.symlink(os.path.join(FLAGS.expdir, 'model'),
               os.path.join(FLAGS.expdir, 'decode', 'model'))

    if FLAGS.computing == 'condor':

        computing_cfg_file = 'config/computing/condor/non_distributed.cfg'
        parsed_computing_cfg = configparser.ConfigParser()
        parsed_computing_cfg.read(computing_cfg_file)
        computing_cfg = dict(parsed_computing_cfg.items('computing'))

        if not os.path.isdir(os.path.join(FLAGS.expdir, 'decode', 'outputs')):
            os.makedirs(os.path.join(FLAGS.expdir, 'decode', 'outputs'))

        subprocess.call(['condor_submit',
                         'expdir=%s' % os.path.join(FLAGS.expdir, 'decode'),
                         'script=nabu/scripts/decode.py',
                         'memory=%s' % computing_cfg['minmemory'],
                         'nabu/computing/condor/non_distributed.job'])


    elif FLAGS.computing == 'standard':
        decode(expdir=os.path.join(FLAGS.expdir, 'decode'))

    else:
        raise Exception('Unknown computing type %s' % FLAGS.computing)

if __name__ == '__main__':
    tf.app.run()

'''dataprep.py
does the data preperation for a single database'''

import sys
import os
sys.path.append(os.getcwd())
import subprocess
import shutil
from six.moves import configparser
import tensorflow as tf
import data

def main(expdir, recipe, computing):
    '''main method'''

    if recipe is None:
        raise Exception('no recipe specified. Command usage: '
                        'nabu data --recipe=/path/to/recipe')
    if not os.path.isdir(recipe):
        raise Exception('cannot find recipe %s' % recipe)
    if expdir is None:
        raise Exception('no expdir specified. Command usage: '
                        'nabu data --expdir=/path/to/recipe '
                        '--recipe=/path/to/recipe')
    if computing not in ['standard', 'condor']:
        raise Exception('unknown computing mode: %s' % computing)

    #read the data conf file
    parsed_cfg = configparser.ConfigParser()
    parsed_cfg.read(os.path.join(recipe, 'database.conf'))

    #loop over the sections in the data config
    for name in parsed_cfg.sections():

        print 'processing %s' % name

        #read the section
        conf = dict(parsed_cfg.items(name))

        #create the expdir for this section
        if not os.path.isdir(os.path.join(expdir, name)):
            os.makedirs(os.path.join(expdir, name))

        #create the database configuration
        dataconf = configparser.ConfigParser()
        dataconf.add_section(name)
        for item in conf:
            dataconf.set(name, item, conf[item])

        with open(os.path.join(expdir, name, 'database.cfg'), 'w') as fid:
            dataconf.write(fid)

        #copy the processor config
        shutil.copyfile(
            conf['processor_config'],
            os.path.join(expdir, name, 'processor.cfg'))

        if computing == 'condor':
            if not os.path.isdir(os.path.join(expdir, name, 'outputs')):
                os.makedirs(os.path.join(expdir, name, 'outputs'))
            subprocess.call(['condor_submit',
                             'expdir=%s' % os.path.join(expdir, name),
                             'nabu/computing/condor/dataprep.job'])
        else:
            data.main(os.path.join(expdir, name))


if __name__ == '__main__':

    tf.app.flags.DEFINE_string('expdir', None,
                               'the exeriments directory'
                              )
    tf.app.flags.DEFINE_string('recipe', None,
                               'The directory containing the recipe')
    tf.app.flags.DEFINE_string('computing', 'standard',
                               'the distributed computing system one of'
                               ' condor'
                              )

    FLAGS = tf.app.flags.FLAGS

    main(FLAGS.expdir, FLAGS.recipe, FLAGS.computing)

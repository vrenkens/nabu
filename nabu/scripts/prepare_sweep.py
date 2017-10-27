'''@file with sweep you can try different parameter sets'''

import os
import shutil
from six.moves import configparser
import tensorflow as tf

def main(sweep, command, expdir, recipe, mode, computing):
    '''main function'''


    #read the sweep file
    with open(sweep) as fid:
        sweeptext = fid.read()

    experiments = [exp.split('\n') for exp in sweeptext.strip().split('\n\n')]
    params = [[param.split() for param in exp[1:]] for exp in experiments]
    expnames = [exp[0] for exp in experiments]

    if not os.path.isdir(expdir):
        os.makedirs(expdir)

    #copy the recipe dir to the expdir
    if os.path.isdir(os.path.join(expdir, 'recipe')):
        shutil.rmtree(os.path.join(expdir, 'recipe'))
    shutil.copytree(recipe, os.path.join(expdir, 'recipe'))

    for i, expname in enumerate(expnames):
        for param in params[i]:
            #read the config
            conf = configparser.ConfigParser()
            conf.read(os.path.join(expdir, 'recipe', param[0]))

            #create the new configuration
            conf.set(param[1], param[2], param[3])
            with open(os.path.join(expdir, 'recipe', param[0]), 'w') as fid:
                conf.write(fid)

        #run the new recipe
        os.system('run %s --expdir=%s --recipe=%s --computing=%s --mode=%s' % (
            command,
            os.path.join(expdir, expname),
            os.path.join(expdir, 'recipe'),
            computing,
            mode
        ))

    #delete the recipe folder
    shutil.rmtree(os.path.join(expdir, 'recipe'))

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('expdir', None,
                               'the exeriments directory'
                              )
    tf.app.flags.DEFINE_string('recipe', None,
                               'The directory containing the recipe'
                              )
    tf.app.flags.DEFINE_string('mode', 'non_distributed',
                               'The computing mode, one of non_distributed, '
                               'single_machine or multi_machine'
                              )
    tf.app.flags.DEFINE_string('computing', 'standart',
                               'the distributed computing system one of'
                               ' standart or condor'
                              )
    tf.app.flags.DEFINE_string('sweep', 'sweep',
                               'the file containing the sweep parameters'
                              )
    tf.app.flags.DEFINE_string('command', 'train',
                               'the command to run'
                              )

    FLAGS = tf.app.flags.FLAGS

    main(FLAGS.sweep, FLAGS.command, FLAGS.expdir, FLAGS.recipe, FLAGS.mode,
         FLAGS.computing)

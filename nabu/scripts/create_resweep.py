'''@file create_resweep.py
create a sweep file that contains the failed experiments'''

import os
import tensorflow as tf

def main(sweep, command, expdir, target):
    '''main function

    Args:
        sweep: the ran sweepfile
        command: the ran command
        expdir: the used experiments directory
        target: the target file
    '''

    #read the sweep file
    with open(sweep) as fid:
        exps = fid.read().split('\n\n')

    #get the name of the directory
    names = [exp.split('\n')[0] for exp in exps]

    #loop over the experiments and write the unfinised ones
    unfinished = 0
    with open(target, 'w') as fid:
        for i, name in enumerate(names):
            if not name:
                continue

            #check if in condor q
            in_queue = os.popen(
                'if condor_q -nobatch -wide | grep -q %s; '
                'then echo true; else echo false; fi' %
                os.path.join(expdir, name)).read().strip() == 'true'

            if not in_queue:

                if command == 'train':

                    #check if training finished
                    if not os.path.exists(
                            os.path.join(expdir, name, 'model', 'checkpoint')):

                        fid.write(exps[i] + '\n')
                        if os.path.exists(os.path.join(
                                expdir, name, 'logdir', 'checkpoint')):

                            fid.write('trainer.cfg trainer'
                                      ' resume_training True\n')

                        fid.write('\n')
                        unfinished += 1

                elif command == 'test':
                    if not os.path.exists(
                            os.path.join(expdir, name, 'test', 'result')):

                        fid.write(exps[i] + '\n\n')
                        unfinished += 1

                else:
                    raise Exception('unknown command %s' % command)

    print '%d unfinished jobs added to %s' % (unfinished, target)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('sweep', None,
                               'the ran sweepfile'
                              )
    tf.app.flags.DEFINE_string('command', None,
                               'the ran command'
                              )
    tf.app.flags.DEFINE_string('expdir', None,
                               'the used experiments directory'
                              )
    tf.app.flags.DEFINE_string('target', None,
                               'the target file'
                              )

    FLAGS = tf.app.flags.FLAGS

    main(
        sweep=FLAGS.sweep,
        command=FLAGS.command,
        expdir=FLAGS.expdir,
        target=FLAGS.target
    )

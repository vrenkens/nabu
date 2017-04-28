'''dataprep.py
does the data preperation for a single database'''

import sys
import os
sys.path.append(os.getcwd())
import gzip
from six.moves import configparser
import tensorflow as tf
from nabu.processing.processors import processor_factory
from nabu.processing.tfwriters import tfwriter_factory

tf.app.flags.DEFINE_string('recipe', None,
                           'The directory containing the recipe')
FLAGS = tf.app.flags.FLAGS

def main(_):
    '''main method'''

    if FLAGS.recipe is None:
        raise Exception('no recipe specified. Command usage: '
                        'nabu data --recipe=/path/to/recipe')
    if not os.path.isdir(FLAGS.recipe):
        raise Exception('cannot find recipe %s' % FLAGS.recipe)

    #read the data conf file
    parsed_cfg = configparser.ConfigParser()
    parsed_cfg.read(os.path.join(FLAGS.recipe, 'database.conf'))

    #loop over the sections in the data config
    for name in parsed_cfg.sections():

        print 'processing %s' % name

        #read the section
        conf = dict(parsed_cfg.items(name))

        if not os.path.exists(conf['dir']):
            os.makedirs(conf['dir'])
        else:
            print '%s already exists, skipping this section' % conf['dir']
            continue

        #read the processor config
        parsed_proc_cfg = configparser.ConfigParser()
        parsed_proc_cfg.read(conf['processor_config'])
        proc_cfg = dict(parsed_proc_cfg.items('processor'))

        #create a processor
        processor = processor_factory.factory(proc_cfg['processor'])(proc_cfg)

        #create a writer
        writer = tfwriter_factory.factory(conf['type'])(conf['dir'])

        #loop over the data files
        for datafile in conf['datafiles'].split(' '):

            if datafile[-3:] == '.gz':
                open_fn = gzip.open
            else:
                open_fn = open

            #loop over the lines in the datafile
            for line in open_fn(datafile):

                #split the name and the data line
                splitline = line.strip().split(' ')
                name = splitline[0]
                dataline = ' '.join(splitline[1:])

                #process the dataline
                processed = processor(dataline)

                #write the processed data to disk
                writer.write(processed, name)

        #write the metadata to file
        processor.write_metadata(conf['dir'])

if __name__ == '__main__':
    tf.app.run()

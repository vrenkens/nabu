'''@file data.py
does the data preperation'''

import os
from six.moves import configparser
import gzip
import tensorflow as tf
from nabu.processing.processors import processor_factory
from nabu.processing.tfwriters import tfwriter_factory

def main(expdir):
    '''main function'''

    #read the data conf file
    parsed_cfg = configparser.ConfigParser()
    parsed_cfg.read(os.path.join(expdir, 'database.cfg'))

    #loop over the sections in the data config
    name = parsed_cfg.sections()[0]

    #read the section
    conf = dict(parsed_cfg.items(name))

    if not os.path.exists(conf['dir']):
        os.makedirs(conf['dir'])
    else:
        print '%s already exists, skipping this section' % conf['dir']
        return

    #read the processor config
    parsed_proc_cfg = configparser.ConfigParser()
    parsed_proc_cfg.read(os.path.join(expdir, 'processor.cfg'))
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
    tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experiments directory')
    FLAGS = tf.app.flags.FLAGS

    main(FLAGS.expdir)

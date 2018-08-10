'''@file default_conf.py
contains functionality for the default configuration'''

import os
from six.moves import configparser

import pdb

def apply_defaults(conf, default_file):
    '''apply the default configuration to conf

    args:
        conf: the configuration as a dictionary that was read in the recipe
        default_file: the file containing the default configurations

    returns: the updated conf
    '''

    if os.path.exists(default_file):

        #read the default configs and put them in a dict
        default = configparser.ConfigParser()
        default.read(default_file)
        default = dict(default.items('default'))

        #go over all the fields in the default and update the conf if it does
        #not have it
        for field in default:
            if field not in conf:
                if default[field] == '':
                    raise Exception(
                        'the field %s was not found in the configuration file'
                        % (field))
                conf[field] = default[field]

    return conf

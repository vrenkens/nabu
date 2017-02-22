'''@file ark.py
contains io functionality for Kaldi ark format'''

import struct
import copy
from collections import OrderedDict
import numpy as np

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)

class ArkReader(object):
    '''
    Class to read Kaldi ark format.
    '''

    def __init__(self, scp_path):
        '''
        ArkReader constructor

        Args:
            scp_path: path to the .scp file
        '''

        self.scp_position = 0
        fin = open(scp_path, "r")
        self.scp_data = OrderedDict()
        line = fin.readline()
        while line != '' and line != None:
            utt_id, path_pos = line.replace('\n', '').split(' ')
            path, pos = path_pos.split(':')
            self.scp_data[utt_id] = (path, pos)
            line = fin.readline()

        fin.close()

    def read_utt_data(self, utt_id):
        '''
        read data from the archive

        Args:
            utt_id: index of the utterance identifier that will be read

        Returns:
            a numpy array containing the data from the utterance
        '''

        ark_read_buffer = open(self.scp_data[utt_id][0], 'rb')
        ark_read_buffer.seek(int(self.scp_data[utt_id][1]), 0)
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != "B":
            print "Input .ark file is not binary"
            exit(1)
        if header[1] == "C":
            print "Input .ark file is compressed"
            exit(1)

        _, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        _, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        if header[1] == "F":
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4),
                                    dtype=np.float32)
        elif header[1] == "D":
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 8),
                                    dtype=np.float64)

        utt_mat = np.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_mat

    def read_next_utt(self):
        '''
        read the next utterance in the scp file

        Returns:
            the utterance ID of the utterance that was read, the utterance data,
            bool that is true if the read utterance was the last one in the file
        '''

        utt_id = self.scp_data.keys()[self.scp_position]
        utt_data = self.read_utt_data(utt_id)

        self.scp_position += 1

        #if at end of file loop around
        if self.scp_position >= len(self.scp_data):
            looped = True
            self.scp_position = 0
        else:
            looped = False

        return (utt_id, utt_data, looped)

    def split(self, num_utt):
        '''take a number of utterances from the ark reader to make a new one

        Args:
            num_utt: the number of utterances in the new ark reader

        Returns:
            an ark reader with the requested number of utterances'''

        reader = copy.deepcopy(self)
        keys = reader.scp_data.keys()[:num_utt]
        reader.scp_data = {key: reader.scp_data[key] for key in keys}
        keys = self.scp_data.keys()[num_utt:]
        self.scp_data = {key: self.scp_data[key] for key in keys}

        return reader

    @property
    def num_utt(self):
        '''the number of utterances in the reader'''
        return len(self.scp_data)

class ArkWriter(object):
    '''
    Class to write to Kaldi ark format
    '''

    def __init__(self, scp_path, default_ark):
        '''
        Arkwriter constructor

        Args:
            scp_path: path to the .scp file that will be written
            default_ark: the name of the default ark file (used when not
                specified)
        '''

        self.scp_path = scp_path
        self.scp_file_write = open(self.scp_path, 'w')
        self.default_ark = default_ark

    def write_next_utt(self, utt_id, utt_mat, ark_path=None):
        '''
        write an utterance to the file

        Args:
            ark_path: path to the .ark file that will be used for writing
            utt_id: the utterance ID
            utt_mat: a numpy array containing the utterance data
        '''

        ark = ark_path or self.default_ark
        ark_file_write = open(ark, 'ab')
        utt_mat = np.asarray(utt_mat, dtype=np.float32)
        rows, cols = utt_mat.shape
        ark_file_write.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        pos = ark_file_write.tell()
        ark_file_write.write(struct.pack('<xcccc', 'B', 'F', 'M', ' '))
        ark_file_write.write(struct.pack('<bi', 4, rows))
        ark_file_write.write(struct.pack('<bi', 4, cols))
        ark_file_write.write(utt_mat)
        self.scp_file_write.write('%s %s:%s\n' % (utt_id, ark, pos))
        ark_file_write.close()

    def close(self):
        '''close the ark writer'''

        self.scp_file_write.close()

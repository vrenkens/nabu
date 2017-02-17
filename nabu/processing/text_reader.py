'''@file textreader.py
contains the Textreader class'''

import os
import numpy as np

class TextReader(object):
    '''reads text from disk'''

    def __init__(self, textfile, max_length, coder, base_pos=0,
                 end_pos=None):
        '''TextReader constructor

        Args:
            textfile: the path to the file containing the text
            max_length: the maximal length of a line
            coder: a TargetCoder object
            base_pos: the base postion where to start reading in the file
            end_pos: optional maximal position in the file'''

        self.max_length = max_length
        self.coder = coder

        #initialise the position to the beginning of the file
        self.base_pos = base_pos
        self.pos = base_pos
        self.end_pos = end_pos or os.path.getsize(textfile)
        if base_pos >= self.end_pos:
            raise Exception('base position should come before end position')

        #store the scp path
        self.textfile = textfile

    def get_utt(self):
        '''read the next line of data specified in the scp file

        Args:
            pos: the desired position in the scp file in bytes

        Returns:
            - the line identifier
            - the read line as a [length x 1] numpy array
            - whether or not the read utterance is the last one
        '''

        #read a line
        line_id, line, looped = self.read_line()

        #encode the line
        encoded = self.coder.encode(line)[:, np.newaxis]

        return line_id, encoded, looped

    def read_line(self):
        '''read the next line of data specified in the scp file

        Args:
            pos: the desired position in the scp file in bytes

        Returns:
            - the line identifier
            - the read line as a string
            - whether or not the read utterance is the last one
        '''

        #create the utteance id
        line_id = 'line%d' % self.pos

        #read a line in the scp file
        with open(self.textfile) as fid:
            fid.seek(self.pos)
            line = fid.readline().strip()
            self.pos = fid.tell()

        #if end of file is reached loop around
        if self.pos >= self.end_pos:
            looped = True
            self.pos = self.base_pos
        else:
            looped = False

        return line_id, line, looped

    def split(self, numlines):
        '''split of a part of the textreader

        Args:
            numlines: number of lines tha should be in the new textreader

        Returns:
            a Textreader object that contains the required number of lines
        '''

        #read the requested number of lines
        self.pos = self.base_pos
        for _ in range(numlines):
            _, _, looped = self.get_utt()
            if looped:
                raise Exception('number of requested lines exeeds the content')

        #create a new textreader with the appropriate boundaries
        textreader = TextReader(self.textfile, self.max_length, self.base_pos,
                                self.pos)

        #update the base position
        self.base_pos = self.pos

        return textreader

    def as_dict(self):
        '''return the reader as a dictionary'''

        #save the position
        pos = self.pos

        #start at the beginning
        self.pos = self.base_pos

        asdict = dict()
        looped = False
        while not looped:
            line_id, line, looped = self.read_line()
            asdict[line_id] = line

        #set the position back to the original
        self.pos = pos

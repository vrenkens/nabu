'''use a segmenting file to segment the audio
usage: python nabu/scripts/segment_audio.py data_dir'''

import os

def main(data_dir):

    wavs = dict()
    with open(os.path.join(data_dir, 'wav.scp')) as fid:
        for line in fid:
            splitline = line.strip().split(' ')
            wavs[splitline[0]] = ' '.join(splitline[1:])

    with open(os.path.join(data_dir, 'segments')) as fid:
        lines = fid.readlines() 

    with open(os.path.join(data_dir, 'segmented.scp'), 'w') as fid:
        for line in lines:
            splitline = line.split(' ')
            fid.write(' '.join([splitline[0], wavs[splitline[1]]] + splitline[2:]))
            
if __name__ == '__main__':
    from inspect import getargspec
    numargs = len(getargspec(main).args)
    if len(sys.argv) != numargs + 1:
        raise Exception(
            'Unexpected number of arguments got %d expected %d'
            % (len(sys.argv) - 1, numargs))
    #pylint: disable=e1120
    main(*sys.argv[1:numargs+1])

    

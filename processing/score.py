'''@file score.py
contains functions to score the system'''

import os
import numpy as np

def cer(decodedir, references):
    '''
    compute the character error rate

    Args:
        decodedir: the directory containing the decoded utterances
        references: the reference transcriptions as a list of strings

    Returns:
    the character error rate
    '''

    errors = 0
    num_labels = 0

    for utt in references:
        #read the best decoded utterance
        if not os.path.exists(decodedir + '/' + utt):
            print 'WARNING: %s was not decoded' % utt
            continue

        nbest = []
        with open(decodedir + '/' + utt) as fid:
            lines = fid.readlines()

        for line in lines:
            splitline = line.strip().split('\t')
            if len(splitline) == 2:
                nbest.append((float(splitline[0]), splitline[1]))
            else:
                nbest.append((float(splitline[0]), ''))

        scores = np.array([h[0] for h in nbest])

        decoded = nbest[np.argmax(scores)][1].split(' ')

        #get the reference as a list
        reference = references[utt].split(' ')

        #compute the error
        error_matrix = np.zeros([len(reference) + 1, len(decoded) + 1])

        error_matrix[:, 0] = np.arange(len(reference) + 1)
        error_matrix[0, :] = np.arange(len(decoded) + 1)

        for x in range(1, len(reference) + 1):
            for y in range(1, len(decoded) + 1):
                error_matrix[x, y] = min([
                    error_matrix[x-1, y] + 1, error_matrix[x, y-1] + 1,
                    error_matrix[x-1, y-1] + (reference[x-1] !=
                                              decoded[y-1])])

        errors += error_matrix[-1, -1]
        num_labels += len(reference)

    return errors/num_labels

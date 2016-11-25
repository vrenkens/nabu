'''@file score.py
contains functions to score the system'''

import numpy as np

def CER(nbests, references):
    '''
    compute the character error rate

    Args:
        nbests: the nbest lists, this is a dictionary with the uttreance id as
            key and a pair containing a list of hypothesis strings and
            a list of log probabilities
        references: the reference transcriptions as a list of strings

    Returns:
    the character error rate
    '''

    errors = 0
    num_labels = 0

    for utt in references:
        #get the single best decoded utterance as a list
        decoded = nbests[utt][0][0].split(' ')

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

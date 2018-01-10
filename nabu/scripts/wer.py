'''
@file wer.py
compute the word error rate with insterions and deletions

usage python wer.py reference decoded
'''

from __future__ import division
import sys
import os
import numpy as np

def main(reference, decoded):
    '''main function

    args:
        reference: the file containing the reference utterances
        decoded: the directory containing the decoded utterances
    '''

    substitutions = 0
    insertions = 0
    deletions = 0
    numwords = 0

    with open(reference) as fid:
        for line in fid:
            #read the reference
            splitline = line.strip().split()
            name = splitline[0]
            reftext = splitline[1:]

            if not os.path.exists(os.path.join(decoded, name)):
                print '%s not decoded, skipping' % name
                continue

            #read the output
            with open(os.path.join(decoded, name)) as did:
                line = did.readline()
            output = line.strip().split()[1:]

            #compare output to reference
            s, i, d = wer(reftext, output)
            substitutions += s
            insertions += i
            deletions += d
            numwords += len(reftext)

    substitutions /= numwords
    deletions /= numwords
    insertions /= numwords
    error = substitutions + deletions + insertions

    print (
        'word error rate: %s\n\tsubstitutions: %s\n\tinsertions: %s\n\t'
        'deletions: %s' % (error, substitutions, insertions, deletions))


def wer(reference, decoded):
    '''
    compute the word error rate

    args:
        reference: a list of the reference words
        decoded: a list of the decoded words

    returns
        - number of substitutions
        - number of insertions
        - number of deletions
    '''

    errors = np.zeros([len(reference) + 1, len(decoded) + 1, 3])
    errors[0, :, 1] = np.arange(len(decoded) + 1)
    errors[:, 0, 2] = np.arange(len(reference) + 1)
    substitution = np.array([1, 0, 0])
    insertion = np.array([0, 1, 0])
    deletion = np.array([0, 0, 1])
    for r, ref in enumerate(reference):
        for d, dec in enumerate(decoded):
            errors[r+1, d+1] = min((
                errors[r, d] + (ref != dec)*substitution,
                errors[r+1, d] + insertion,
                errors[r, d+1] + deletion), key=np.sum)

    return tuple(errors[-1, -1])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

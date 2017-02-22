'''@file score.py
contains functions to score the system'''

import numpy as np

def cer(outputs, targets):
    '''
    compute the character error rate

    Args:
        outputs: a dictionary containing the decoder outputs
        targets: a dictionary containing the reference outputs

    Returns:
    the character error rate
    '''

    errors = 0
    num_labels = 0

    for utt in targets:
        scores = np.array([h[0] for h in outputs[utt]])

        decoded = outputs[utt][np.argmax(scores)][1].split(' ')

        #get the reference as a list
        reference = targets[utt].split(' ')

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

        print 'decoded: %s' % ' '.join(decoded)
        print 'reference: %s' % ' '.join(reference)
        print 'score: %f' % (float(error_matrix[-1, -1])/len(reference))

    return errors/num_labels

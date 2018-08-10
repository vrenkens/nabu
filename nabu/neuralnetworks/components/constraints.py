'''@file constraints.py
contains constraints for the model weights'''

import tensorflow as tf

class MaxNorm(object):
    '''a constraint for a maximum norm of the weights'''

    def __init__(self, max_norm=1, axis=0):
        '''
        constructor

        args:
            max_norm: the maximum norm
            axis: the axis to compute the norm
        '''

        self._axis = axis
        self._max_norm = max_norm

    def __call__(self, tensor):
        '''apply the constraint

        args:
            tensor: the tensor to apply the constraint to'''

        with tf.name_scope('MaxNorm'):
            norms = tf.norm(tensor, axis=self._axis)
            clipped = tf.clip_by_value(norms, 0, 1)
            out = tensor*clipped/(norms + tensor.dtype.min)

        return out

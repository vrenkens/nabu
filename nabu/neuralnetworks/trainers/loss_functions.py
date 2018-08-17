'''@file loss_functions.py
contains functions to compute the training loss'''

import tensorflow as tf
from nabu.neuralnetworks.components import ops

def factory(loss_function):
    '''factory method for the loss function

    args:
        loss_function: the required loss function

    returns:
        a callable loss function
    '''

    if loss_function == 'sum_cross_entropy':
        return sum_cross_entropy
    elif loss_function == 'average_cross_entropy':
        return average_cross_entropy
    elif loss_function == 'CTC':
        return CTC
    elif loss_function == 'average_sigmoid_cross_entropy':
        return average_sigmoid_cross_entropy
    elif loss_function == 'marigin':
        return marigin_loss
    else:
        raise Exception('unknown loss function %s' % loss_function)

def marigin_loss(targets, logits, logit_seq_length, target_seq_length):
    '''
    marigin loss

    Args:
        targets: a dictionary of [batch_size x time x ...] tensor containing
            the targets
        logits: a dictionary of [batch_size x time x ...] tensor containing
            the logits
        logit_seq_length: a dictionary of [batch_size] vectors containing
            the logit sequence lengths
        target_seq_length: a dictionary of [batch_size] vectors containing
            the target sequence lengths

    Returns:
        a scalar value containing the loss
    '''

    with tf.name_scope('marigin_loss'):
        losses = []

        for t in targets:

            #compute probs
            probs = tf.nn.sigmoid(logits[t])

            #compute the lower and upper marigins
            lower = tf.square(tf.maximum(0.0, probs - 0.1))
            upper = tf.square(tf.maximum(0.0, 0.9 - probs))

            #compute the loss
            tar = tf.to_float(targets)
            loss = tar*upper + (1-tar)*lower

            #mask the loss
            loss = tf.where(
                tf.sequence_mask(logit_seq_length[t], tf.shape(targets[t])[1]),
                loss,
                tf.zeros_like(loss))


            losses.append(tf.reduce_mean(
                tf.reduce_sum(loss)/tf.expand_dims(logit_seq_length[t], 1)))

        loss = tf.reduce_sum(losses)

    return loss

def cross_entropy(targets, logits, seq_length):
    '''
    compute the cross entropy for all sequences in the batch

    Args:
        targets: a dictionary of [batch_size x time x ...] tensor containing
            the targets
        logits: a dictionary of [batch_size x time x ...] tensor containing
            the logits
        seq_length: a dictionary of [batch_size] vectors containing
            the sequence lengths

    Returns:
        a dictionarie of vectors of [batch_size] containing the cross_entropy
    '''

    losses = {}

    for t in targets:

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits[t],
            labels=tf.cast(targets[t], tf.int32))

        loss = tf.where(
            tf.sequence_mask(seq_length[t], tf.shape(targets[t])[1]),
            loss,
            tf.zeros_like(loss))

        losses[t] = tf.reduce_sum(loss, 1)

    return losses

def sigmoid_cross_entropy(targets, logits, seq_length):
    '''
    compute the sigmnoid cross entropy for all sequences in the batch

    Args:
        targets: a dictionary of [batch_size x time x ...] tensor containing
            the targets
        logits: a dictionary of [batch_size x time x ...] tensor containing
            the logits
        seq_length: a dictionary of [batch_size] vectors containing
            the sequence lengths

    Returns:
        a dictionary of vectors of [batch_size] containing the cross_entropy
    '''

    losses = {}

    for t in targets:

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits[t],
            labels=targets[t])

        loss = tf.where(
            tf.sequence_mask(seq_length[t], tf.shape(targets[t])[1]),
            loss,
            tf.zeros_like(loss))

        losses[t] = tf.reduce_sum(loss, 1)

    return losses

def sum_cross_entropy(targets, logits, logit_seq_length, target_seq_length):
    '''cross entropy summed over timesteps'''

    with tf.name_scope('sum_cross_entropy_loss'):
        losses = cross_entropy(
            targets, logits, target_seq_length)
        losses = {t: tf.reduce_mean(losses[t]) for t in losses}
        loss = tf.reduce_sum(losses.values())

    return loss

def average_cross_entropy(targets, logits, logit_seq_length, target_seq_length):
    '''cross entropy averaged over timesteps'''

    with tf.name_scope('average_cross_entropy_loss'):
        losses = cross_entropy(
            targets, logits, logit_seq_length)
        losses = {t: tf.reduce_mean(losses[t]/tf.to_float(target_seq_length[t]))
                  for t in losses}
        loss = tf.reduce_sum(losses.values())

    return loss

def average_sigmoid_cross_entropy(
        targets, logits, logit_seq_length, target_seq_length):
    '''sigmoid cross entropy averaged over timesteps'''

    with tf.name_scope('average_cross_entropy_loss'):
        losses = sigmoid_cross_entropy(
            targets, logits, logit_seq_length)
        losses = {t: tf.reduce_mean(losses[t]/tf.to_float(target_seq_length[t]))
                  for t in losses}
        loss = tf.reduce_sum(losses.values())

    return loss

def CTC(targets, logits, logit_seq_length, target_seq_length):
    '''
    CTC loss

    Args:
        targets: a dictionary of [batch_size x time x ...] tensor containing
            the targets
        logits: a dictionary of [batch_size x time x ...] tensor containing
            the logits
        logit_seq_length: a dictionary of [batch_size] vectors containing
            the logit sequence lengths
        target_seq_length: a dictionary of [batch_size] vectors containing
            the target sequence lengths

    Returns:
        a scalar value containing the loss
    '''
    with tf.name_scope('CTC_loss'):

        losses = []

        for t in targets:
            #convert the targets into a sparse tensor representation
            sparse_targets = ops.dense_sequence_to_sparse(
                targets[t], target_seq_length[t])

            losses.append(tf.reduce_mean(tf.nn.ctc_loss(
                sparse_targets,
                logits[t],
                logit_seq_length[t],
                time_major=False)))

        loss = tf.reduce_sum(losses)

    return loss

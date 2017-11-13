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

    if loss_function == 'cross_entropy':
        return cross_entropy
    elif loss_function == 'cross_entropy_eos':
        return cross_entropy_eos
    elif loss_function == 'CTC':
        return CTC
    elif loss_function == 'sigmoid_cross_entropy':
        return sigmoid_cross_entropy
    else:
        raise Exception('unknown loss function %s' % loss_function)

def cross_entropy(targets, logits, logit_seq_length, target_seq_length):
    '''
    cross enthropy loss

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

    with tf.name_scope('cross_entropy_loss'):
        losses = []

        for t in targets:
            #stack the logits
            stacked_logits = ops.seq2nonseq(logits[t], logit_seq_length[t])

            #create the stacked targets
            stacked_targets = ops.seq2nonseq(targets[t],
                                             target_seq_length[t])
            stacked_targets = tf.cast(stacked_targets, tf.int32)

            losses.append(tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=stacked_logits,
                    labels=stacked_targets)))

        loss = tf.reduce_sum(losses)

    return loss

def cross_entropy_eos(targets, logits, logit_seq_length, target_seq_length):
    '''
    cross enthropy loss with an end of sequence label added

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

    with tf.name_scope('cross_entropy_loss'):
        losses = []
        batch_size = tf.shape(targets.values()[0])[0]

        for t in targets:
            with tf.name_scope('cross_entropy_loss'):

                output_dim = tf.shape(logits[t])[2]

                #get the logits for the final timestep
                indices = tf.stack([tf.range(batch_size),
                                    logit_seq_length[t] - 1],
                                   axis=1)
                final_logits = tf.gather_nd(logits[t], indices)

                #stack all the logits except the final logits
                stacked_logits = ops.seq2nonseq(logits[t],
                                                logit_seq_length[t] - 1)

                #create the stacked targets
                stacked_targets = ops.seq2nonseq(targets[t],
                                                 target_seq_length[t])

                #create the targets for the end of sequence labels
                final_targets = tf.tile([output_dim-1], [batch_size])

                #add the final logits and targets
                stacked_logits = tf.concat([stacked_logits, final_logits], 0)
                stacked_targets = tf.concat([stacked_targets, final_targets], 0)

                #compute the cross-entropy loss
                losses.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=stacked_logits,
                        labels=stacked_targets)))

        loss = tf.reduce_sum(losses)

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

def sigmoid_cross_entropy(targets, logits, logit_seq_length, target_seq_length):
    '''
    Sigmoid cross entropy

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

    with tf.name_scope('sigmoid_cross_entropy_loss'):
        losses = []

        for t in targets:
            #stack all the logits except the final logits
            stacked_logits = ops.seq2nonseq(logits[t], logit_seq_length[t])

            #create the stacked targets
            stacked_targets = ops.seq2nonseq(
                tf.cast(targets[t], tf.float32),
                target_seq_length[t])

            losses.append(tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=stacked_logits,
                    labels=stacked_targets)))

        loss = tf.reduce_sum(losses)

    return loss

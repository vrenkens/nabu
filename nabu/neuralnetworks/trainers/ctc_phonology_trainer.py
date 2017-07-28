'''@file ctctrainer.py
contains the CTCPhonologyTrainer'''

import numpy as np
import tensorflow as tf
import trainer
from nabu.neuralnetworks.components import ops

class CTCPhonologyTrainer(trainer.Trainer):
    '''A trainer that minimises the CTC loss of a phonological feature extractor

    the phonological features are first mapped to phones, the mapping depends
    on the language
    '''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-entropy loss for every input
        frame and ads an end of sequence label to the targets

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
        with tf.name_scope('phonological_CTC_loss'):

            #read the mappings for all the languages
            langs = self.conf['languages'].split(' ')
            mappings = []
            for lang in langs:
                with open(self.conf['mapping_%s' % lang], 'rb') as fid:

                    #load the mapping
                    mapping = np.load(fid)

                    #add the blank mapping
                    blank_mappings = []
                    for f in logits:
                        numfeats = int(logits[f].get_shape()[2])
                        blank_mapping = np.zeros([numfeats, 1])
                        blank_mapping[-1, 0] = 1
                        blank_mappings.append(blank_mapping)
                    blank_mapping = np.concatenate(blank_mappings, 0)
                    mapping = np.concatenate([mapping, blank_mapping], 1)

                    mappings.append(mapping)

            mappings = np.stack(mappings, axis=0)
            mappings = tf.constant(mappings, dtype=tf.float32)

            #get the features
            feature_names = self.model.output_names

            #concatenate all the feature outputs
            feat_logits = tf.concat([logits[f] for f in feature_names], axis=2)
            seq_length = logit_seq_length.values()[0]

            #get the appropriate mappings for all the utterances
            batch_mappings = tf.gather(mappings, targets['lang'][:, 0])

            #map the feature logits to phone logits
            phone_logits = tf.matmul(feat_logits, batch_mappings)

            #convert the targets into a sparse tensor representation
            sparse_targets = ops.dense_sequence_to_sparse(
                targets['phones'], target_seq_length['phones'])

            #compute the loss
            loss = tf.reduce_sum(tf.nn.ctc_loss(
                sparse_targets,
                phone_logits,
                seq_length,
                time_major=False))

        return loss

    @property
    def trainlabels(self):
        '''
        the number of aditional labels the trainer needs (e.g. blank or eos)
        '''

        return 1

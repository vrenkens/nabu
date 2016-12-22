'''@file nnet.py
contains the functionality for a Kaldi style neural network'''

import os
from classifiers import *
import tensorflow as tf
from trainer import CTCTrainer
from decoder import CTCDecoder
from math import ceil
import numpy as np

class Nnet(object):
    '''a class for using a DBLTSM with CTC for ASR'''

    def __init__(self, conf, savedir, input_dim, num_labels):
        '''
        Nnet constructor

        Args:
            conf: nnet configuration
            savedir: the directory where everything will be saved
            input_dim: network input dimension
            num_labels: number of target labels
        '''

        #save the conf
        self.conf = conf

        #define location to save neural nets
        self.savedir = savedir

        if not os.path.isdir(savedir + '/training'):
            os.makedirs(savedir + '/training')
        if not os.path.isdir(savedir + '/validation'):
            os.makedirs(savedir + '/validation')

        #save the input dim
        self.input_dim = input_dim

        #create a DBLSTM
        class_name = '%s.%s' % (conf['module'], conf['class'])
        self.classifier = eval(class_name)(conf, num_labels + 1)


    def train(self, dispenser, val_dispenser):
        '''
        Train the neural network

        Args:
            dispenser: a batchdispenser for training
            val_dispenser: a batchdispenser used for validation
        '''

        #compute the total number of steps
        num_steps = int(dispenser.num_batches *int(self.conf['num_epochs']))

        #create the trainer
        print 'building the training graph'
        trainer = CTCTrainer(
            classifier=self.classifier,
            input_dim=self.input_dim,
            max_input_length=dispenser.max_input_length,
            max_target_length=dispenser.max_target_length,
            init_learning_rate=float(self.conf['initial_learning_rate']),
            learning_rate_decay=float(self.conf['learning_rate_decay']),
            num_steps=num_steps,
            batch_size=int(self.conf['batch_size']),
            numbatches_to_aggregate=int(self.conf['numbatches_to_aggregate']),
            logdir=self.savedir + '/logdir',
            beam_width=int(self.conf['beam_width']))

        #start the trainer
        trainer.start()
        step = 0

        #start the training iteration
        while step < num_steps:

            #get a batch of data
            batch_data, batch_labels = dispenser.get_batch()

            #update the model
            loss, lr = trainer.update(batch_data, batch_labels)

            #print the progress
            print ('step %d/%d loss: %f, learning rate: %f'
                   %(step, num_steps, loss, lr))

            #increment the step
            step += 1

        #save the final model
        trainer.save_model(self.savedir + '/final')

        #stop the trainer
        trainer.stop()

    def decode(self, reader, target_coder):
        '''
        compute pseudo likelihoods the testing set

        Args:
            reader: a feature reader object to read features to decode
            target_coder: target coder object to decode the target sequences

        Returns:
            a dictionary with the utterance id as key and a pair as Value
            containing:
                - a list of hypothesis strings
                - a numpy array of log probabilities for the hypotheses
        '''

        #create a decoder
        print 'building the decoding graph'
        decoder = CTCDecoder(self.classifier, self.input_dim,
                             reader.max_input_length,
                             int(self.conf['beam_width']))


        #start tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101

        nbests = dict()

        with tf.Session(graph=decoder.graph, config=config):

            #load the model
            decoder.restore(self.savedir + '/final')

            #feed the utterances one by one to the neural net
            while True:
                utt_id, utt_mat, looped = reader.get_utt()

                if looped:
                    break

                #compute predictions
                encoded_hypotheses, logprobs = decoder(utt_mat)

                #decode the hypotheses
                hypotheses = [target_coder.decode(h)
                              for h in encoded_hypotheses]

                nbests[utt_id] = (hypotheses, logprobs)


        return nbests

    def validate(self, dispenser, trainer):
        '''validate the performance of the model

        Args:
            dispenser: the batchdispenser for the validation data
            trainer: the trainer object

        Returns:
            the average validation loss
        '''

        losses = np.zeros(0)

        for _ in range(int(ceil(dispenser.num_batches))):

            #get a batch of data
            val_data, val_labels = dispenser.get_batch(True)
            num_utt = len(val_data)

            #pad the data with empty utterances untill batch size
            val_data = (
                val_data + (dispenser.size - num_utt)*
                [np.zeros([0,val_data[0].shape[1]])])
            val_labels = (
                val_labels + (dispenser.size - num_utt)*
                [np.zeros([0])])

            #get the losses for the this batch of data
            batch_losses = trainer.evaluate(val_data, val_labels)
            losses = np.append(losses, batch_losses[:num_utt])

        validation_loss = losses.mean()

        return validation_loss

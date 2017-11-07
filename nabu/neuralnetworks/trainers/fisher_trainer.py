'''@file fisher_trainer.py
contains the FisherTrainer'''

import os
import StringIO
import ConfigParser
import tensorflow as tf
from nabu.neuralnetworks.trainers import trainer
from nabu.neuralnetworks.decoders import random_decoder
from nabu.processing import input_pipeline
from nabu.neuralnetworks.components import hooks
import trainer_factory

class FisherTrainer(trainer.Trainer):
    '''A that computes the fisher information matrix at the end of training'''

    def __init__(self,
                 conf,
                 dataconf,
                 modelconf,
                 evaluatorconf,
                 expdir,
                 server,
                 task_index):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            conf: the trainer config as a ConfigParser
            dataconf: the data configuration as a ConfigParser
            modelconf: the neural net model configuration
            evaluatorconf: the evaluator configuration for evaluating
                if None no evaluation will be done
            expdir: directory where the summaries will be written
            server: optional server to be used for distributed training
            task_index: optional index of the worker task in the cluster
        '''

        # Create a deep copy of the conf
        config_string = StringIO.StringIO()
        conf.write(config_string)

        # We must reset the buffer to make it ready for reading.
        config_string.seek(0)
        wrapped_conf = ConfigParser.ConfigParser()
        wrapped_conf.readfp(config_string)

        #set the wrapped section as the trainer section
        for option, value in wrapped_conf.items(conf.get('trainer', 'wrapped')):
            wrapped_conf.set('trainer', option, value)
        wrapped_conf.remove_section(conf.get('trainer', 'wrapped'))

        #create the wrapped trainer
        self.wrapped = trainer_factory.factory(
            wrapped_conf.get('trainer', 'trainer'))(
                wrapped_conf,
                dataconf,
                modelconf,
                evaluatorconf,
                expdir,
                server,
                task_index)

        #super constructor
        super(FisherTrainer, self).__init__(
            conf,
            dataconf,
            modelconf,
            evaluatorconf,
            expdir,
            server,
            task_index
        )

        #link the models of the wrapped trainer to this trainer
        self.wrapped.model = self.model

        #the scope for the fisher information
        self.fisher_scope = tf.VariableScope(
            tf.AUTO_REUSE, 'fisher_information')

        self.decoderconf = conf

    @property
    def fisher(self):
        '''
        get the fisher information matrices and initial values

        returns:
            - a dict containing the fisher information
            - a dict containing the initial values
        '''

        variables = []
        if self.conf['fisher_encoder'] == 'True':
            variables += self.model.encoder.variables
        if self.conf['fisher_decoder'] == 'True':
            variables += self.model.decoder.variables

        fisher = {}
        init = {}
        with tf.variable_scope(self.fisher_scope):
            for var in variables:
                scope = os.path.join(*os.path.split(var.name)[:-1])
                name = os.path.split(var.op.name)[-1]
                with tf.variable_scope(scope):
                    with tf.variable_scope('fisher_information'):
                        fisher[var] = tf.get_variable(
                            name=name,
                            shape=var.shape,
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer(),
                            trainable=False
                        )
                    with tf.variable_scope('initial_value'):
                        init[var] = tf.get_variable(
                            name=name,
                            shape=var.shape,
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer(),
                            trainable=False
                        )

        return fisher, init

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

        #get the fisher information and initial values
        fisher, init = self.fisher

        #compute the fisher loss
        with tf.variable_scope('fisher_loss'):
            fisher_loss = float(self.conf['fisher_weight'])*tf.reduce_sum([
                tf.reduce_sum(fisher[var]*tf.square(var-init[var]))
                for var in fisher])

        return self.wrapped.compute_loss(targets, logits, logit_seq_length,
                                         target_seq_length) + fisher_loss

    @property
    def trainlabels(self):
        '''
        the number of aditional labels the trainer needs (e.g. blank or eos)
        '''

        return self.wrapped.trainlabels

    def chief_only_hooks(self, outputs):
        '''add hooks only for the chief worker

        Args:
            outputs: the outputs generated by the create graph method

        Returns:
            a list of hooks
        '''

        hooklist = self.wrapped.chief_only_hooks(outputs)

        #create a hook to compute the fisher information
        fisher, init = self.fisher
        variables = (fisher.values() + init.values())

        if variables:

            #create an input pipeline
            input_names = self.model.conf.get('io', 'inputs').split(' ')
            if input_names == ['']:
                input_names = []
            input_sections = [self.conf['fisher_' + i].split(' ')
                              for i in input_names]
            input_dataconfs = {}
            for i, sectionset in enumerate(input_sections):
                input_dataconfs[input_names[i]] = []
                for section in sectionset:
                    input_dataconfs[input_names[i]].append(
                        dict(self.dataconf.items(section)))

            #create the random sample decoder
            decoder = random_decoder.RandomDecoder(self.decoderconf, self.model)

            hooklist.append(ComputeFisher(
                fisher,
                input_dataconfs,
                decoder))

            fisher_file = os.path.join(self.conf['fisher_dir'], 'fisher.ckpt')

            #if there is fisher information to be loaded create a hook for it
            if os.path.exists(fisher_file):
                hooklist.append(hooks.LoadAtBegin(
                    fisher_file,
                    variables))

            #create a hook to save the fisher information at the end
            if not os.path.isdir(self.conf['fisher_dir']):
                os.makedirs(self.conf['fisher_dir'])
            hooklist.append(hooks.SaveAtEnd(
                fisher_file,
                variables))

        return hooklist

    def hooks(self, outputs):
        '''add hooks for the session

        Args:
            outputs: the outputs generated by the create graph method

        Returns:
            a list of hooks
        '''

        return self.wrapped.hooks(outputs)

class ComputeFisher(tf.train.SessionRunHook):
    '''a hook to compute the fisher information matrix'''

    def __init__(
            self,
            fisher,
            dataconfs,
            decoder,):
        '''
        hook constructor

        Args:
            outputs: the outputs genereated by create_graph
        '''

        self.fisher = fisher
        self.dataconfs = dataconfs
        self.decoder = decoder

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201

        with tf.variable_scope('compute_fisher'):

            data_queue_elements, _ = input_pipeline.get_filenames(
                self.dataconfs.values())

            self.num_samples = len(data_queue_elements)

            data_queue = tf.train.string_input_producer(
                string_tensor=data_queue_elements,
                shuffle=False,
                seed=None,
                capacity=1)

            inputs, input_seq_length, _ = input_pipeline.input_pipeline(
                data_queue=data_queue,
                batch_size=1,
                numbuckets=1,
                dataconfs=self.dataconfs.values(),
                variable_batch_size=False)

            inputs = {
                self.dataconfs.keys()[i]: d
                for i, d in enumerate(inputs)}
            input_seq_length = {
                self.dataconfs.keys()[i]: d
                for i, d in enumerate(input_seq_length)}

            #get the input log likelihood using the random sample decoder
            logprob = self.decoder(inputs, input_seq_length).values()[0][2][0]

            #get the derivative the logprob
            gradients = tf.gradients(logprob, self.fisher.keys())

            #create an op to update the fisher information
            update_ops = []
            for var, grad in zip(self.fisher.keys(), gradients):
                update_ops.append(
                    self.fisher[var].assign_add(
                        tf.square(grad)/self.num_samples).op)
            self.update_fisher = tf.group(*update_ops)

    def end(self, session):
        '''this will be run at session closing'''

        #compute the fisher information matrix
        print 'computing fisher information'
        for _ in range(self.num_samples):
            session.run(self.update_fisher)

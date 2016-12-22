'''@file trainer.py
neural network trainer environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from classifiers import seq_convertors

import pdb

class Trainer(object):
    '''General class outlining the training environment of a classifier.'''
    __metaclass__ = ABCMeta

    def __init__(self, classifier, input_dim, max_input_length,
                 max_target_length, init_learning_rate, learning_rate_decay,
                 num_steps, batch_size, numbatches_to_aggregate, logdir):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            classifier: the neural net classifier that will be trained
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximal length of the input sequences
            max_target_length: the maximal length of the target sequences
            init_learning_rate: the initial learning rate
            learning_rate_decay: the parameter for exponential learning rate
                decay
            num_steps: the total number of steps that will be taken
            batch_size: determines how many utterances are
                processed at a time to limit memory usage
            cluster: the optional cluster used for distributed training, it
                should contain at least one parmeter server and one worker
            numbatches_to_aggregate: number of batches that are aggragated
                before the gradients are applied
            logdir: directory where the summaries will be written
        '''

        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        #create the graph
        self.graph = tf.Graph()

        #define the placeholders in the graph
        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[batch_size, max_input_length, input_dim],
                name='inputs')

            #reference labels
            self.targets = tf.placeholder(
                tf.int32, shape=[batch_size, max_target_length, 1],
                name='targets')


            #the length of all the input sequences
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[batch_size], name='input_seq_length')

            #the length of all the output sequences
            self.target_seq_length = tf.placeholder(
                tf.int32, shape=[batch_size], name='output_seq_length')

            #compute the training outputs of the classifier
            trainlogits, logit_seq_length, self.modelsaver, self.control_ops =\
                classifier(
                    self.inputs, self.input_seq_length, targets=self.targets,
                    target_seq_length=self.target_seq_length, is_training=True,
                    reuse=False, scope='Classifier')

            #compute the validation output of the classifier
            logits, _, _, _ = classifier(
                self.inputs, self.input_seq_length, targets=self.targets,
                target_seq_length=self.target_seq_length, is_training=False,
                reuse=True, scope='Classifier')


            with tf.variable_scope('train'):

                #a variable to hold the amount of steps already taken
                self.global_step = tf.get_variable(
                    'global_step', [], dtype=tf.int32,
                    initializer=tf.constant_initializer(0), trainable=False)

                #a variable to scale the learning rate (used to reduce the
                #learning rate in case validation performance drops)
                learning_rate_fact = tf.get_variable(
                    'learning_rate_fact', [],
                    initializer=tf.constant_initializer(1.0), trainable=False)

                #operation to half the learning rate
                self.halve_learningrate_op = learning_rate_fact.assign(
                    learning_rate_fact/2).op

                #compute the learning rate with exponential decay and scale with
                #the learning rate factor
                self.learning_rate = tf.train.exponential_decay(
                    init_learning_rate, self.global_step, num_steps,
                    learning_rate_decay) * learning_rate_fact

                #create the optimizer
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                #create an optimizer that aggregates gradients
                optimizer = tf.train.SyncReplicasOptimizerV2(
                    opt=optimizer,
                    replicas_to_aggregate=numbatches_to_aggregate,
                    total_num_replicas=1)

                #compute the loss
                self.loss = self.compute_loss(
                    self.targets, trainlogits, logit_seq_length,
                    self.target_seq_length)

                #compute the gradients
                grads = optimizer.compute_gradients(self.loss)

                with tf.variable_scope('clip'):
                    #clip the gradients
                    grads = [(tf.clip_by_value(grad, -1., 1.), var)
                             for grad, var in grads]

                #opperation to apply the gradients
                apply_gradients_op = optimizer.apply_gradients(
                    grads, global_step=self.global_step, name='apply_gradients')

                #all remaining operations with the UPDATE_OPS GraphKeys
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                #create an operation to update the gradients, the batch_loss
                #and do all other update ops
                #pylint: disable=E1101
                self.update_op = tf.group(*([apply_gradients_op] + update_ops),
                                          name='update')

            with tf.name_scope('valid'):
                #compute the outputs that will be used for validation
                self.outputs = self.validation(logits, logit_seq_length)

            # add an operation to initialise all the variables in the graph
            self.init_op = tf.initialize_all_variables()

            #operations to initialise the token queue
            self.init_token_op = optimizer.get_init_tokens_op()
            self.chief_queue_runner = optimizer.get_chief_queue_runner()

            #create the summaries for visualisation
            tf.summary.scalar('loss', self.loss)

            #saver for the training network
            self.trainsaver = tf.train.Saver()

        #create the supervisor
        self.supervisor = tf.train.Supervisor(
            graph=self.graph,
            ready_for_local_init_op=optimizer.ready_for_local_init_op,
            is_chief=True,
            init_op=self.init_op,
            local_init_op=optimizer.chief_init_op,
            logdir=logdir,
            saver=self.trainsaver,
            global_step=self.global_step,
            save_summaries_secs=0)

        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

        #start with an empty session
        self.session = None

    @abstractmethod
    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the loss, this is specific to each
        trainer

        Args:
            targets: a list that contains a Bx1 tensor containing the targets
                for eacht time step where B is the batch size
            logits: a list that contains a BxO tensor containing the output
                logits for eacht time step where O is the output dimension
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the output sequences as a
                vector

        Returns:
            a scalar value containing the total loss
        '''

        raise NotImplementedError('Abstract method')

    @abstractmethod
    def validation(self, logits, logit_seq_length):
        '''
        compute the outputs that will be used for validation

        Args:
            logits: the classifier output logits
            logit_seq_length: the sequence lengths of the logits

        Returns:
            The validation outputs
        '''

        raise NotImplementedError('Abstract method')

    @abstractmethod
    def validation_metric(self, outputs, targets):
        '''
        compute the metric that will be used for validation

        The metric can be e.g. an error rate or a loss

        Args:
            outputs: the validation outputs
            targets: the ground truth labels

        Returns:
            The validation outputs
        '''

        raise NotImplementedError('Abstract method')

    def start(self):
        '''do all the needed preperations for training and start the session'''

        #start the session and standart servises
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        self.session = self.supervisor.PrepareSession(config=config)

        #start the queue runners
        self.supervisor.start_queue_runners(
            sess=self.session,
            queue_runners=[self.chief_queue_runner])

        #fill the queue with initial tokens
        self.session.run(self.init_token_op)

    def stop(self):
        '''close the session'''

        self.session.close()

    def update(self, inputs, targets):
        '''
        update the neural model with a batch or training data

        Args:
            inputs: the inputs to the neural net, this should be a list
                containing an NxF matrix for each utterance in the batch where
                N is the number of frames in the utterance
            targets: the targets for neural nnet, this should be
                a list containing an N-dimensional vector for each utterance

        Returns:
            a pair containing:
                - the loss at this step
                - the learning rate used at this step
        '''
        if self.session is None:
            raise Exception('Trainer not started')

        #get a list of sequence lengths
        input_seq_length = [i.shape[0] for i in inputs]
        target_seq_length = [t.shape[0] for t in targets]

        #pad the inputs and targets till the maximum lengths
        padded_inputs = np.array(pad(inputs, self.max_input_length))
        padded_targets = np.array(pad(targets, self.max_target_length))
        padded_targets = padded_targets[:,:,np.newaxis]

        #pylint: disable=E1101
        _, loss, summary, lr = self.session.run(
            fetches=[self.update_op,
                     self.loss,
                     self.supervisor.summary_op,
                     self.learning_rate],
            feed_dict={self.inputs:padded_inputs,
                       self.targets:padded_targets,
                       self.input_seq_length:input_seq_length,
                       self.target_seq_length:target_seq_length})

        #add the summary
        self.supervisor.summary_writer.add_summary(
            summary=summary,
            global_step=self.global_step.eval(session=self.session))

        return loss, lr

    def evaluate(self, inputs, targets):
        '''
        Evaluate the performance of the neural net

        Args:
            inputs: the inputs to the neural net, this should be a list
                containing NxF matrices for each utterance in the batch where
                N is the number of frames in the utterance
            targets: the one-hot encoded targets for neural nnet, this should be
                a list containing an NxO matrix for each utterance where O is
                the output dimension of the neural net

        Returns:
            a numpy array containing the loss of each utterance in the batch
        '''

        if self.session is None:
            raise Exception('Trainer not started')

        if inputs is None or targets is None:
            return None

        #get a list of sequence lengths
        input_seq_length = [i.shape[0] for i in inputs]

        #pad the inputs and targets till the maximum lengths
        padded_inputs = np.array(pad(inputs, self.max_input_length))

        #pylint: disable=E1101
        outputs = list(self.outputs.eval(
            session=self.session,
            feed_dict={self.inputs:padded_inputs,
                       self.input_seq_length:input_seq_length}))

        #compute the validation error
        error = self.validation_metric(outputs[:len(targets)], targets)

        return error

    def halve_learning_rate(self):
        '''halve the learning rate'''

        self.halve_learningrate_op.run(session=self.session)

    def save_model(self, filename):
        '''
        Save the model

        Args:
            filename: path to the model file
        '''
        self.modelsaver.save(self.session, filename)

    def restore_model(self, filename):
        '''
        Load the model

        Args:
            filename: path where the model will be saved
        '''
        self.modelsaver.restore(self.session, filename)

    def save_trainer(self, filename):
        '''
        Save the training progress (including the model)

        Args:
            filename: path where the model will be saved
        '''
        self.trainsaver.save(self.session, filename)

    def restore_trainer(self, filename):
        '''
        Load the training progress (including the model)

        Args:
            filename: path where the model will be saved
        '''
        self.trainsaver.restore(self.session, filename)

class CrossEnthropyTrainer(Trainer):
    '''A trainer that minimises the cross-enthropy loss, the output sequences
    must be of the same length as the input sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a list that contains a Bx1 tensor containing the targets
                for eacht time step where B is the batch size
            logits: a list that contains a BxO tensor containing the output
                logits for eacht time step where O is the output dimension
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the target sequences as a
                vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_enthropy_loss'):

            #convert to non sequential data
            nonseq_targets = seq_convertors.seq2nonseq(targets,
                                                       target_seq_length)
            nonseq_logits = seq_convertors.seq2nonseq(logits, logit_seq_length)

            #make a vector out of the targets
            nonseq_targets = tf.reshape(nonseq_targets, [-1])

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets,
                                        int(nonseq_logits.get_shape()[1]))

            #compute the cross-enthropy loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                nonseq_logits, nonseq_targets))

        return loss

    def validation(self, logits, logit_seq_length):
        '''
        apply a softmax to the logits so the cross-enthropy can be computed

        Args:
            logits: a [batch_size, max_input_length, dim] tensor containing the
                logits
            logit_seq_length: the length of all the input sequences as a vector

        Returns:
            a tensor with the same shape as logits with the label probabilities
        '''

        return tf.nn.softmax(logits)

    def validation_metric(self, outputs, targets):
        '''the cross-enthropy

        Args:
            outputs: the validation output, which is a matrix containing the
                label probabilities of size [batch_size, max_input_length, dim].
            targets: a list containing the ground truth target labels
        '''

        loss = np.zeros(outputs.shape[0])

        for utt in range(outputs.shape[0]):
            loss[utt] += np.mean(-np.log(
                outputs[utt, np.arange(targets[utt].size), targets[utt]]))

        return loss

class CTCTrainer(Trainer):
    '''A trainer that minimises the CTC loss, the output sequences'''
    def __init__(self, classifier, input_dim, max_input_length,
                 max_target_length, init_learning_rate, learning_rate_decay,
                 num_steps, batch_size, numbatches_to_aggregate, logdir,
                 beam_width):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            classifier: the neural net classifier that will be trained
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximal length of the input sequences
            max_target_length: the maximal length of the target sequences
            init_learning_rate: the initial learning rate
            learning_rate_decay: the parameter for exponential learning rate
                decay
            num_steps: the total number of steps that will be taken
            batch_size: the batch size
            numbatches_to_aggregate: number of batches that are aggragated
                before the gradients are applied
            logdir: directory where the summaries will be written
            beam_width: the width of the beam used for validation
        '''

        self.beam_width = beam_width

        super(CTCTrainer, self).__init__(
            classifier=classifier,
            input_dim=input_dim,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            init_learning_rate=init_learning_rate,
            learning_rate_decay=learning_rate_decay,
            num_steps=num_steps,
            batch_size=batch_size,
            numbatches_to_aggregate=numbatches_to_aggregate,
            logdir=logdir)

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the CTC loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a [batch_size, max_target_length, 1] tensor containing the
                targets
            logits: a [batch_size, max_input_length, dim] tensor containing the
                inputs
            logit_seq_length: the length of all the input sequences as a vector
            target_seq_length: the length of all the target sequences as a
                vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('CTC_loss'):

            #get the batch size
            batch_size = int(targets.get_shape()[0])

            #convert the targets into a sparse tensor representation
            indices = tf.concat(0, [tf.concat(
                1, [tf.expand_dims(tf.tile([s], [target_seq_length[s]]), 1),
                    tf.expand_dims(tf.range(target_seq_length[s]), 1)])
                                    for s in range(batch_size)])

            values = tf.reshape(
                seq_convertors.seq2nonseq(targets, target_seq_length), [-1])

            shape = [batch_size, int(targets.get_shape()[1])]

            sparse_targets = tf.SparseTensor(tf.cast(indices, tf.int64), values,
                                             shape)

            loss = tf.reduce_mean(tf.nn.ctc_loss(logits, sparse_targets,
                                                logit_seq_length,
                                                time_major=False))

        return loss

    def validation(self, logits, logit_seq_length):
        '''
        decode the validation set with CTC beam search

        Args:
            logits: a [batch_size, max_input_length, dim] tensor containing the
                inputs
            logit_seq_length: the length of all the input sequences as a vector

        Returns:
            a matrix containing the decoded labels with size
            [batch_size, max_decoded_length]
        '''

        #Convert logits to time major
        tm_logits = tf.transpose(logits, [1, 0, 2])

        #do the CTC beam search
        sparse_output, _ = tf.nn.ctc_beam_search_decoder(
            tf.pack(tm_logits), logit_seq_length, self.beam_width)

        #convert the output to dense tensors with -1 as default values
        dense_output = tf.sparse_tensor_to_dense(sparse_output[0],
                                                 default_value=-1)

        return dense_output


    def validation_metric(self, outputs, targets):
        '''the Label Error Rate for the decoded labels

        Args:
            outputs: the validation output, which is a matrix containing the
                decoded labels of size [batch_size, max_decoded_length]. the
                output sequences are padded with -1
            targets: a list containing the ground truth target labels
        '''

        #remove the padding from the outputs
        trimmed_outputs = [o[np.where(o != -1)] for o in outputs]

        ler = np.zeros(len(targets))

        for k, target in enumerate(targets):

            error_matrix = np.zeros([target.size + 1,
                                     trimmed_outputs[k].size + 1])

            error_matrix[:, 0] = np.arange(target.size + 1)
            error_matrix[0, :] = np.arange(trimmed_outputs[k].size + 1)

            for x in range(1, target.size + 1):
                for y in range(1, trimmed_outputs[k].size + 1):
                    error_matrix[x, y] = min([
                        error_matrix[x-1, y] + 1, error_matrix[x, y-1] + 1,
                        error_matrix[x-1, y-1] + (target[x-1] !=
                                                  trimmed_outputs[k][y-1])])

            ler[k] = error_matrix[-1, -1]/target.size

        return ler

def pad(inputs, length):
    '''
    Pad the inputs so they have the maximum length

    Args:
        inputs: the inputs, this should be a list containing time major
            tenors
        length: the length that will be used for padding the inputs

    Returns:
        the padded inputs
    '''
    padded_inputs = [np.append(
        i, np.zeros([length-i.shape[0]] + list(i.shape[1:])), 0)
                     for i in inputs]

    return padded_inputs

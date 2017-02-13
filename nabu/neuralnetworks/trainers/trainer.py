'''@file trainer.py
neural network trainer environment'''

import os
import shutil
from abc import ABCMeta, abstractmethod
from time import time, sleep
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
import numpy as np
from nabu.neuralnetworks.decoders import decoder_factory

class Trainer(object):
    '''General class outlining the training environment of a classifier.'''
    __metaclass__ = ABCMeta

    def __init__(self,
                 conf,
                 decoder_conf,
                 classifier,
                 input_dim,
                 max_input_length,
                 max_target_length,
                 dispenser,
                 val_reader=None,
                 val_targets=None,
                 expdir=None,
                 server=None,
                 cluster=None,
                 task_index=0):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            classifier: the neural net classifier that will be trained
            conf: the trainer config
            decoder_conf: the decoder config used for validation
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximal length of the input sequences
            max_target_length: the maximal length of the target sequences
            num_steps: the total number of steps that will be taken
            dispenser: a Batchdispenser object
            cluster: the optional cluster used for distributed training, it
                should contain at least one parmeter server and one worker
            val_reader: the feature reader for the validation data if None
                validation will not be used
            val_targets: a dictionary containing the targets of the validation
                set
            expdir: directory where the summaries will be written
            server: optional server to be used for distributed training
            cluster: optional cluster to be used for distributed training
            task_index: optional index of the worker task in the cluster
        '''

        self.conf = conf
        self.dispenser = dispenser
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.num_steps = int(dispenser.num_batches*int(conf['num_epochs'])
                             /max(1, int(conf['numbatches_to_aggregate'])))
        self.val_reader = val_reader
        self.val_targets = val_targets
        self.expdir = expdir
        self.server = server
        self.cluster = cluster

        #create the graph
        self.graph = tf.Graph()

        if cluster is None:
            #non distributed training
            is_chief = True
            num_replicas = 1

            #choose a GPU if it is available otherwise take a CPU
            local_device_protos = (list_local_devices())
            gpu = [x.name for x in local_device_protos
                   if x.device_type == 'GPU']
            if len(gpu) > 0:
                device = str(gpu[0])
            else:
                device = str(local_device_protos[0].name)

        else:
            #distributed training
            num_replicas = len(cluster.as_dict()['worker'])
            numservers = len(cluster.as_dict()['ps'])
            is_chief = task_index == 0
            device = tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)

        #define the placeholders in the graph
        with self.graph.as_default():

            with tf.device(device):

                #create the inputs placeholder
                self.inputs = tf.placeholder(
                    dtype=tf.float32,
                    shape=[dispenser.size, max_input_length, input_dim],
                    name='inputs')

                #reference labels
                self.targets = tf.placeholder(
                    dtype=tf.int32,
                    shape=[dispenser.size, max_target_length],
                    name='targets')

                #the length of all the input sequences
                self.input_seq_length = tf.placeholder(
                    dtype=tf.int32,
                    shape=[dispenser.size],
                    name='input_seq_length')

                #the length of all the output sequences
                self.target_seq_length = tf.placeholder(
                    dtype=tf.int32,
                    shape=[dispenser.size],
                    name='output_seq_length')

                #a placeholder to set the position
                self.pos_in = tf.placeholder(
                    dtype=tf.int32,
                    shape=[],
                    name='pos_in')

                self.val_loss_in = tf.placeholder(
                    dtype=tf.float32,
                    shape=[],
                    name='val_loss_in')

                #compute the training outputs of the classifier
                trainlogits, logit_seq_length = classifier(
                    inputs=self.inputs,
                    input_seq_length=self.input_seq_length,
                    targets=self.targets,
                    target_seq_length=self.target_seq_length,
                    is_training=True)

                #create a saver for the classifier
                self.modelsaver = tf.train.Saver(tf.trainable_variables())

                #create a decoder object for validation
                self.decoder = decoder_factory.factory(
                    conf=decoder_conf,
                    classifier=classifier,
                    input_dim=input_dim,
                    max_input_length=max_input_length,
                    coder=dispenser.target_coder,
                    expdir=expdir,
                    decoder_type=decoder_conf['decoder'])

                if cluster is not None:
                    #create a done queue for each parameter server
                    done_queues = []
                    for ps in range(numservers):
                        with tf.device("/job:ps/task:%d" % (ps)):
                            done_queues.append(tf.FIFOQueue(
                                capacity=num_replicas,
                                dtypes=tf.int32,
                                shared_name='done_queue' + str(ps)))

                    #create an operation that enqueues an element in each done
                    #queue
                    self.done = tf.group(*[q.enqueue(0) for q in done_queues])

                else:
                    self.done = tf.no_op()

                with tf.variable_scope('train'):
                    #a variable to hold the amount of steps already taken
                    self.global_step = tf.get_variable(
                        'global_step', [], dtype=tf.int32,
                        initializer=tf.constant_initializer(0), trainable=False)

                    #a variable that indicates if features are being read
                    self.reading = tf.get_variable(
                        name='reading',
                        shape=[],
                        dtype=tf.bool,
                        initializer=tf.constant_initializer(False),
                        trainable=False)

                    #the position in the feature reader
                    self.pos = tf.get_variable(
                        name='position',
                        shape=[],
                        dtype=tf.int32,
                        initializer=tf.constant_initializer(0),
                        trainable=False)

                    #the current validation loss
                    self.val_loss = tf.get_variable(
                        name='validation_loss',
                        shape=[],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(1.79e+308),
                        trainable=False)

                    #a variable that specifies when the model was last validated
                    self.validated_step = tf.get_variable(
                        name='validated_step',
                        shape=[],
                        dtype=tf.int32,
                        initializer=tf.constant_initializer(
                            -int(conf['valid_frequency'])),
                        trainable=False)

                    #operation to start reading
                    self.block_reader = self.reading.assign(True).op

                    #operation to release the reader
                    self.release_reader = self.reading.assign(False).op

                    #operation to set the position
                    self.set_pos = self.pos.assign(self.pos_in).op

                    #operation to update the validated steps
                    self.set_val_step = self.validated_step.assign(
                        self.global_step).op

                    #operation to set the validation loss
                    self.set_val_loss = self.val_loss.assign(
                        self.val_loss_in).op

                    #a variable to scale the learning rate (used to reduce the
                    #learning rate in case validation performance drops)
                    learning_rate_fact = tf.get_variable(
                        name='learning_rate_fact',
                        shape=[],
                        initializer=tf.constant_initializer(1.0),
                        trainable=False)

                    #operation to half the learning rate
                    self.halve_learningrate_op = learning_rate_fact.assign(
                        learning_rate_fact/2).op

                    #compute the learning rate with exponential decay and scale
                    #with the learning rate factor
                    self.learning_rate = (tf.train.exponential_decay(
                        learning_rate=float(conf['initial_learning_rate']),
                        global_step=self.global_step,
                        decay_steps=self.num_steps,
                        decay_rate=float(conf['learning_rate_decay']))
                                          * learning_rate_fact)

                    #create the optimizer
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)

                    #create an optimizer that aggregates gradients
                    if int(conf['numbatches_to_aggregate']) > 0:
                        optimizer = tf.train.SyncReplicasOptimizerV2(
                            opt=optimizer,
                            replicas_to_aggregate=int(
                                conf['numbatches_to_aggregate']),
                            total_num_replicas=num_replicas)

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
                        grads_and_vars=grads,
                        global_step=self.global_step,
                        name='apply_gradients')

                    #all remaining operations with the UPDATE_OPS GraphKeys
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    #create an operation to update the gradients, the batch_loss
                    #and do all other update ops
                    #pylint: disable=E1101
                    self.update_op = tf.group(
                        *([apply_gradients_op] + update_ops),
                        name='update')

                # add an operation to initialise all the variables in the graph
                self.init_op = tf.global_variables_initializer()

                #operations to initialise the token queue
                if int(conf['numbatches_to_aggregate']) > 0:
                    self.init_token_op = optimizer.get_init_tokens_op()
                    self.chief_queue_runner = optimizer.get_chief_queue_runner()
                    ready_for_local_init_op = optimizer.ready_for_local_init_op
                    if is_chief:
                        local_init_op = optimizer.chief_init_op
                    else:
                        local_init_op = optimizer.local_step_init_op
                else:
                    self.init_token_op = tf.no_op()
                    self.chief_queue_runner = None
                    ready_for_local_init_op = tf.no_op()
                    local_init_op = tf.no_op()

                #create the summaries for visualisation
                tf.summary.scalar('validation loss', self.val_loss)
                tf.summary.scalar('learning rate', self.learning_rate)

                #create a histogram for all trainable parameters
                for param in tf.trainable_variables():
                    tf.summary.histogram(param.name, param)


                #saver for the training network
                self.trainsaver = tf.train.Saver()

        #create the supervisor
        self.supervisor = tf.train.Supervisor(
            graph=self.graph,
            ready_for_local_init_op=ready_for_local_init_op,
            is_chief=is_chief,
            init_op=self.init_op,
            local_init_op=local_init_op,
            logdir=self.expdir + '/logdir',
            saver=self.trainsaver,
            global_step=self.global_step)

        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

    @abstractmethod
    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the loss, this is specific to each
        trainer

        Args:
            targets: a [batch_size, max_target_length] tensor containing the
                targets
            logits: a [batch_size, max_logit_length, dim] tensor containing the
                logits
            logit_seq_length: the length of all the logit sequences as a
                [batch_size] vector
            target_seq_length: the length of all the target sequences as a
                [batch_size] vector

        Returns:
            a scalar value containing the total loss
        '''

        raise NotImplementedError('Abstract method')

    def train(self):
        '''train the model'''

        #look for the master if distributed training is done
        if self.server is None:
            master = ''
        else:
            master = self.server.target

        #start the session and standart servises
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        config.allow_soft_placement = True

        with self.supervisor.managed_session(master, config=config) as sess:

            #start the queue runners
            if self.chief_queue_runner is not None:
                self.supervisor.start_queue_runners(
                    sess=sess,
                    queue_runners=[self.chief_queue_runner])

            #fill the queue with initial tokens
            sess.run(self.init_token_op)

            #set the reading flag to false
            sess.run(self.release_reader)

            #start the training loop
            with self.supervisor.stop_on_exception():
                while (not self.supervisor.should_stop()
                       and self.global_step.eval(sess) < self.num_steps):

                    #check if validation is due
                    [step, val_step] = sess.run(
                        [self.global_step, self.validated_step])
                    if (step - val_step >= int(self.conf['valid_frequency'])
                            and int(self.conf['valid_frequency']) > 0):

                        self.validate(sess)

                    #start time
                    start = time()

                    #wait until the reader is free
                    while self.reading.eval(sess):
                        sleep(1)

                    #block the reader
                    sess.run(self.block_reader)

                    #read a batch of data
                    batch_data, batch_labels = self.dispenser.get_batch(
                        self.pos.eval(sess))

                    #update the position
                    self.set_pos.run(
                        session=sess,
                        feed_dict={self.pos_in:self.dispenser.pos})

                    #release the reader
                    sess.run(self.release_reader)

                    #update the model
                    loss, lr = self.update(batch_data, batch_labels, sess)

                    print(('step %d/%d loss: %f, learning rate: %f, '
                           'time elapsed: %f sec')
                          %(self.global_step.eval(sess), self.num_steps,
                            loss, lr, time()-start))

                #the chief will create the final model
                if self.supervisor.is_chief:
                    modeldir = self.expdir + '/model'

                    if not os.path.isdir(modeldir):
                        os.mkdir(modeldir)

                    #save the network
                    self.modelsaver.save(sess, modeldir + '/network.ckpt')

                    #copy the needed config files
                    shutil.copyfile(self.expdir + '/features.cfg',
                                    modeldir + '/features.cfg')
                    shutil.copyfile(self.expdir + '/asr.cfg',
                                    modeldir + '/asr.cfg')
                    shutil.copyfile(self.expdir + '/decoder.cfg',
                                    modeldir + '/decoder.cfg')

                    #write the used alphabet in the model directory
                    alphabet = self.dispenser.target_coder.lookup.keys()
                    with open(modeldir + '/alphabet', 'w') as fid:
                        for target in alphabet:
                            fid.write(target + '\n')

                    #write the maximium input length to the model file
                    with open(modeldir + '/max_input_length', 'w') as fid:
                        fid.write(str(self.max_input_length))

                #notify the parameter server that he worker has terminated
                sess.run(self.done)

                self.supervisor.request_stop()

    def update(self, inputs, targets, sess):
        '''
        update the neural model with a batch or training data

        Args:
            inputs: the inputs to the neural net, this should be a list
                containing an NxF matrix for each utterance in the batch where
                N is the number of frames in the utterance
            targets: the targets for neural net, this should be
                a list containing an N-dimensional vector for each utterance
            sess: the session

        Returns:
            a pair containing:
                - the loss at this step
                - the learning rate used at this step
        '''

        #get a list of sequence lengths
        input_seq_length = [i.shape[0] for i in inputs]
        target_seq_length = [t.shape[0] for t in targets]

        #pad the inputs and targets untill the maximum lengths
        padded_inputs = np.array(pad(inputs, self.max_input_length))
        padded_targets = np.array(pad(targets, self.max_target_length))

        #pylint: disable=E1101
        _, loss, lr = sess.run(
            fetches=[self.update_op,
                     self.loss,
                     self.learning_rate],
            feed_dict={self.inputs:padded_inputs,
                       self.targets:padded_targets,
                       self.input_seq_length:input_seq_length,
                       self.target_seq_length:target_seq_length})

        return loss, lr

    def validate(self, sess):
        '''
        Evaluate the performance of the neural net and halves the learning rate
        if it is worse

        Args:
            inputs: the inputs to the neural net, this should be a list
                containing NxF matrices for each utterance in the batch where
                N is the number of frames in the utterance
            targets: the one-hot encoded targets for neural net, this should be
            a list containing an NxO matrix for each utterance where O is
                the output dimension of the neural net

        '''

        #update the validated step
        sess.run([self.set_val_step])

        outputs = self.decoder.decode(self.val_reader, sess)

        val_loss = self.decoder.score(outputs, self.val_targets)

        print 'validation loss: %f' % val_loss

        if (val_loss > self.val_loss.eval(session=sess)
                and self.conf['valid_adapt'] == 'True'):
            print 'halving learning rate'
            sess.run([self.halve_learningrate_op])

        sess.run(self.set_val_loss, feed_dict={self.val_loss_in:val_loss})

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

def wait(server, task_index, numworkers):
    '''wait for the workers to finish

    Args:
        server: the Tensorflow server
        task_index: the ps task_index
        numworkers: total number of workers
    '''

    print 'waiting for workers to finish'

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/job:ps/task:%d" % (task_index)):
            done_queue = tf.FIFOQueue(
                capacity=numworkers,
                dtypes=tf.int32,
                shared_name='done_queue' + str(task_index))
            dequeue_op = done_queue.dequeue()

    graph.finalize()

    with tf.Session(target=server.target, graph=graph) as sess:
        for i in range(numworkers):
            sess.run(dequeue_op)
            print '%d/%d workers have finished' % (i+1, numworkers)

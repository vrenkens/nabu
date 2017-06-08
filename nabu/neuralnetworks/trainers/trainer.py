'''@file trainer.py
neural network trainer environment'''

import os
from abc import ABCMeta, abstractmethod, abstractproperty
from time import time
import cPickle as pickle
import tensorflow as tf
from nabu.processing import input_pipeline
from nabu.neuralnetworks.models.model import Model
from nabu.neuralnetworks.evaluators import evaluator_factory
from nabu.neuralnetworks.components.hooks import SaveAtEnd

class Trainer(object):
    '''General class outlining the training environment of a model.'''
    __metaclass__ = ABCMeta

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
            conf: the trainer config
            dataconf: the data configuration as a ConfigParser
            modelconf: the neural net model configuration
            evaluatorconf: the evaluator configuration for evaluating
                if None no evaluation will be done
            expdir: directory where the summaries will be written
            server: optional server to be used for distributed training
            task_index: optional index of the worker task in the cluster
        '''

        self.expdir = expdir
        self.server = server
        self.conf = conf

        cluster = tf.train.ClusterSpec(server.server_def.cluster)

        #create the graph
        self.graph = tf.Graph()

        #get the database configurations
        input_names = modelconf.get('io', 'inputs').split(' ')
        if input_names == ['']:
            input_names = []
        input_sections = [conf[i] for i in input_names]
        input_dataconfs = []
        for section in input_sections:
            input_dataconfs.append(dict(dataconf.items(section)))
        output_names = modelconf.get('io', 'outputs').split(' ')
        if output_names == ['']:
            output_names = []
        target_sections = conf['targets'].split(' ')
        target_dataconfs = []
        for section in target_sections:
            target_dataconfs.append(dict(dataconf.items(section)))

        #create the model
        modelfile = os.path.join(expdir, 'model', 'model.pkl')
        with open(modelfile, 'wb') as fid:
            self.model = Model(
                conf=modelconf,
                targetconfs=target_dataconfs,
                trainlabels=self.trainlabels)
            pickle.dump(self.model, fid)

        #create the evaluator
        evaltype = evaluatorconf.get('evaluator', 'evaluator')
        if evaltype != 'None':
            evaluator = evaluator_factory.factory(evaltype)(
                conf=evaluatorconf,
                dataconf=dataconf,
                model=self.model
            )

        if 'local' in cluster.as_dict():
            num_replicas = 1
            device = tf.DeviceSpec(job='local')
        else:
            #distributed training
            num_replicas = len(cluster.as_dict()['worker'])
            num_servers = len(cluster.as_dict()['ps'])
            ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
                num_tasks=num_servers,
                load_fn=tf.contrib.training.byte_size_load_fn
            )
            device = tf.train.replica_device_setter(
                ps_tasks=num_servers,
                ps_strategy=ps_strategy)
            chief_ps = tf.DeviceSpec(
                job='ps',
                task=0)

        self.is_chief = task_index == 0

        #define the placeholders in the graph
        with self.graph.as_default():

            #create a local num_steps variable
            self.num_steps = tf.get_variable(
                name='num_steps',
                shape=[],
                dtype=tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False
            )

            with tf.device(device):

                #check if running in distributed model
                if 'local' in cluster.as_dict():

                    #get the filenames
                    data_queue_elements, _ = input_pipeline.get_filenames(
                        input_dataconfs + target_dataconfs)

                    #create the data queue and queue runners
                    data_queue = tf.train.string_input_producer(
                        string_tensor=data_queue_elements,
                        shuffle=True,
                        seed=None,
                        capacity=int(conf['batch_size'])*2,
                        shared_name='data_queue')

                    #compute the number of steps
                    num_steps = (int(conf['num_epochs'])*
                                 len(data_queue_elements)/
                                 int(conf['batch_size']))

                    #set the number of steps
                    self.set_num_steps = self.num_steps.assign(num_steps).op
                    self.done = tf.no_op()

                else:
                    with tf.device(chief_ps):

                        #get the data queue
                        data_queue = tf.FIFOQueue(
                            capacity=int(conf['batch_size'])*(num_replicas+1),
                            shared_name='data_queue',
                            name='data_queue',
                            dtypes=[tf.string],
                            shapes=[[]])

                        #get the number of steps from the parameter server
                        num_steps_queue = tf.FIFOQueue(
                            capacity=1,
                            dtypes=[tf.int32],
                            shared_name='num_steps_queue',
                            name='num_steps_queue',
                            shapes=[[]]
                        )

                        #set the number of steps
                        self.set_num_steps = self.num_steps.assign(
                            num_steps_queue.dequeue()).op

                    #get the done queues
                    done_ops = []
                    for i in range(num_servers):
                        with tf.device('job:ps/task:%d' % i):
                            done_queue = tf.FIFOQueue(
                                capacity=1,
                                dtypes=[tf.bool],
                                shapes=[[]],
                                shared_name='done_queue%d' % task_index,
                                name='done_queue%d' % task_index
                            )

                            done_ops.append(done_queue.enqueue(True))

                    self.done = tf.group(*done_ops)

                #a variable to hold the amount of steps already taken
                self.global_step = tf.get_variable(
                    name='global_step',
                    shape=[],
                    dtype=tf.int32,
                    initializer=tf.constant_initializer(0),
                    trainable=False)

                #create a check if training should continue
                self.should_stop = tf.greater_equal(self.global_step,
                                                    self.num_steps)

                #training part
                with tf.variable_scope('train'):

                    #create the input pipeline
                    data, seq_length = input_pipeline.input_pipeline(
                        data_queue=data_queue,
                        batch_size=int(conf['batch_size']),
                        numbuckets=int(conf['numbuckets']),
                        dataconfs=input_dataconfs + target_dataconfs
                    )

                    inputs = {
                        input_names[i]: d
                        for i, d in enumerate(data[:len(input_sections)])}
                    input_seq_length = {
                        input_names[i]: d
                        for i, d in enumerate(seq_length[:len(input_sections)])}
                    targets = {
                        output_names[i]: d
                        for i, d in enumerate(data[len(input_sections):])}
                    target_seq_length = {
                        output_names[i]: d
                        for i, d in enumerate(seq_length[len(input_sections):])}

                    #store the inputs for creating the decoding graph
                    self.inputs = inputs
                    self.input_seq_length = input_seq_length

                    #compute the training outputs of the model
                    logits, logit_seq_length = self.model(
                        inputs=inputs,
                        input_seq_length=input_seq_length,
                        targets=targets,
                        target_seq_length=target_seq_length,
                        is_training=True)

                    #a variable to scale the learning rate (used to reduce the
                    #learning rate in case validation performance drops)
                    learning_rate_fact = tf.get_variable(
                        name='learning_rate_fact',
                        shape=[],
                        initializer=tf.constant_initializer(1.0),
                        trainable=False)

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
                        optimizer = tf.train.SyncReplicasOptimizer(
                            opt=optimizer,
                            replicas_to_aggregate=int(
                                conf['numbatches_to_aggregate']),
                            total_num_replicas=num_replicas)


                    #compute the loss
                    self.loss = self.compute_loss(
                        targets, logits, logit_seq_length, target_seq_length)

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
                    self.update_op = tf.group(
                        *([apply_gradients_op] + update_ops),
                        name='update')

                if self.is_chief and evaltype != 'None':

                    #validation part
                    with tf.variable_scope('validate'):

                        #create a variable to hold the validation loss
                        self.validation_loss = tf.get_variable(
                            name='validation_loss',
                            shape=[],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(1.79e+308),
                            trainable=False)

                        #create an op to reset the validation loss
                        self.reset_val_loss = self.validation_loss.assign(0).op

                        #create a variable to save the last step where the model
                        #was validated
                        validated_step = tf.get_variable(
                            name='validated_step',
                            shape=[],
                            dtype=tf.int32,
                            initializer=tf.constant_initializer(
                                -int(conf['valid_frequency'])),
                            trainable=False)

                        #a check if validation is due
                        self.should_validate = tf.greater_equal(
                            self.global_step - validated_step,
                            int(conf['valid_frequency']))

                        #compute the loss
                        val_batch_loss, self.valbatches = evaluator.evaluate()

                        self.update_loss = self.validation_loss.assign(
                            self.validation_loss +
                            val_batch_loss/self.valbatches
                        ).op

                        #update the learning rate factor
                        self.half_lr = learning_rate_fact.assign(
                            learning_rate_fact/2).op

                        #create an operation to updated the validated step
                        self.update_validated_step = validated_step.assign(
                            self.global_step).op

                        tf.summary.scalar('validation loss',
                                          self.validation_loss)
                else:
                    self.update_loss = None

                tf.summary.scalar('learning rate', self.learning_rate)

                #create a histogram for all trainable parameters
                for param in tf.trainable_variables():
                    tf.summary.histogram(param.name, param)

                #create the scaffold
                self.scaffold = tf.train.Scaffold()


    @abstractmethod
    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a list of [batch_size x ...] tensor containing the
                targets
            logits: a list of [batch_size x ... tensor containing the
                logits
            logit_seq_length: a list of [batch_size] vectors containing the
                logit sequence lengths
            target_seq_length: a list of [batch_size] vectors containing the
                target sequence lengths

        Returns:
            a scalar value containing the loss
        '''

    @abstractproperty
    def trainlabels(self):
        '''
        the number of aditional labels the trainer needs (e.g. blank or eos)
        '''

    def train(self):
        '''train the model'''

        #look for the master if distributed training is done
        master = self.server.target

        #start the session and standart servises
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True

        #create a hook for saving the final model
        save_hook = SaveAtEnd(
            os.path.join(self.expdir, 'model', 'network.ckpt'),
            self.model)

        with self.graph.as_default():
            with tf.train.MonitoredTrainingSession(
                master=master,
                is_chief=self.is_chief,
                checkpoint_dir=os.path.join(self.expdir, 'logdir'),
                scaffold=self.scaffold,
                chief_only_hooks=[save_hook],
                config=config) as sess:

                #set the number of steps
                self.set_num_steps.run(session=sess)

                #start the training loop
                #pylint: disable=E1101
                while not (sess.should_stop() or
                           self.should_stop.eval(session=sess)):

                    #check if validation is due
                    if (self.update_loss is not None
                            and self.should_validate.eval(session=sess)):

                        #get the previous validation loss
                        prev_val_loss = self.validation_loss.eval(
                            session=sess)

                        #set the validated step
                        self.update_validated_step.run(session=sess)

                        #reset the validation loss
                        self.reset_val_loss.run(session=sess)

                        #compute the validation loss
                        for _ in range(self.valbatches):
                            self.update_loss.run(session=sess)

                        #get the current validation loss
                        validation_loss = self.validation_loss.eval(
                            session=sess)

                        print 'validation loss: %f' % validation_loss

                        #check if the validation loss is better
                        if (validation_loss > prev_val_loss and
                                self.conf['valid_adapt'] == 'True'):

                            print ('validation loss is worse halving '
                                   'learning rate')
                            self.half_lr.run(session=sess)

                    #start time
                    start = time()

                    #update the model
                    _, loss, lr, global_step, num_steps = sess.run(
                        fetches=[self.update_op,
                                 self.loss,
                                 self.learning_rate,
                                 self.global_step,
                                 self.num_steps])

                    print(('step %d/%d loss: %f, learning rate: %f, '
                           'time elapsed: %f sec')
                          %(global_step,
                            num_steps,
                            loss, lr, time()-start))

                self.done.run(session=sess)

class ParameterServer(object):
    '''a class for parameter servers'''

    def __init__(self,
                 conf,
                 modelconf,
                 dataconf,
                 server,
                 task_index):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            conf: the trainer config
            modelconf: the model configuration
            dataconf: the data configuration as a ConfigParser
            server: optional server to be used for distributed training
            task_index: optional index of the worker task in the cluster
        '''

        self.graph = tf.Graph()
        self.server = server
        self.task_index = task_index

        #distributed training
        cluster = tf.train.ClusterSpec(server.server_def.cluster)
        num_replicas = len(cluster.as_dict()['worker'])

        with self.graph.as_default():

            #the chief parameter server should create the data queue
            if task_index == 0:
                #get the database configurations
                inputs = modelconf.get('io', 'inputs').split(' ')
                if inputs == ['']:
                    inputs = []
                input_sections = [conf[i] for i in inputs]
                input_dataconfs = []
                for section in input_sections:
                    input_dataconfs.append(dict(dataconf.items(section)))
                outputs = modelconf.get('io', 'outputs').split(' ')
                if outputs == ['']:
                    outputs = []
                target_sections = [conf[o] for o in outputs]
                target_dataconfs = []
                for section in target_sections:
                    target_dataconfs.append(dict(dataconf.items(section)))

                data_queue_elements, _ = input_pipeline.get_filenames(
                    input_dataconfs + target_dataconfs)

                tf.train.string_input_producer(
                    string_tensor=data_queue_elements,
                    shuffle=True,
                    seed=None,
                    capacity=int(conf['batch_size'])*(num_replicas+1),
                    shared_name='data_queue')

                num_steps = (int(conf['num_epochs'])*len(data_queue_elements)/
                             int(conf['batch_size']))

                #create a queue to communicate the number of steps
                num_steps_queue = tf.FIFOQueue(
                    capacity=num_replicas,
                    dtypes=[tf.int32],
                    shapes=[[]],
                    shared_name='num_steps_queue',
                    name='num_steps_queue'
                )

                self.set_num_steps = num_steps_queue.enqueue_many(
                    tf.constant([num_steps]*num_replicas)
                )


            #create a queue for the workers to signiy that they are done
            done_queue = tf.FIFOQueue(
                capacity=num_replicas,
                dtypes=[tf.bool],
                shapes=[[]],
                shared_name='done_queue%d' % task_index,
                name='done_queue%d' % task_index
            )

            self.wait_op = done_queue.dequeue_many(num_replicas).op

            self.scaffold = tf.train.Scaffold()

    def join(self):
        '''wait for the workers to finish'''

        with self.graph.as_default():
            with tf.train.MonitoredTrainingSession(
                master=self.server.target,
                is_chief=self.task_index == 0,
                scaffold=self.scaffold) as sess:

                if self.task_index == 0:
                    self.set_num_steps.run(session=sess)
                self.wait_op.run(session=sess)

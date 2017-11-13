'''@file trainer.py
neural network trainer environment'''

import os
import time
import cPickle as pickle
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.python.client import device_lib
from nabu.processing import input_pipeline
from nabu.neuralnetworks.trainers import loss_functions
from nabu.neuralnetworks.models.model import Model
from nabu.neuralnetworks.evaluators import evaluator_factory
from nabu.neuralnetworks.components import hooks

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
            conf: the trainer config as a ConfigParser
            dataconf: the data configuration as a ConfigParser
            modelconf: the neural net model configuration
            evaluatorconf: the evaluator configuration for evaluating
                if None no evaluation will be done
            expdir: directory where the summaries will be written
            server: optional server to be used for distributed training
            task_index: optional index of the worker task in the cluster
        '''

        #save some inputs
        self.conf = dict(conf.items('trainer'))
        self.dataconf = dataconf
        self.evaluatorconf = evaluatorconf
        self.expdir = expdir
        self.server = server
        self.task_index = task_index

        #create the model
        self.model = Model(
            conf=modelconf,
            trainlabels=int(self.conf['trainlabels']))

    def _create_graph(self):
        '''
        create the trainer computational graph

        Returns:
            - a dictionary of graph outputs
        '''

        cluster = tf.train.ClusterSpec(self.server.server_def.cluster)

        #the outputs of the graph
        outputs = {}

        device, chief_ps = self._device(cluster)

        #a variable to hold the amount of steps already taken
        outputs['global_step'] = tf.get_variable(
            name='global_step',
            shape=[],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0),
            trainable=False)

        should_terminate = tf.get_variable(
            name='should_terminate',
            shape=[],
            dtype=tf.bool,
            initializer=tf.constant_initializer(False),
            trainable=False)

        outputs['terminate'] = should_terminate.assign(True).op

        #create an op that measures the memory usage
        if [x for x in device_lib.list_local_devices()
                if x.device_type == 'GPU']:

            outputs['memory_usage']\
                = tf.contrib.memory_stats.MaxBytesInUse()
            outputs['memory_limit'] = tf.contrib.memory_stats.BytesLimit()
        else:
            outputs['memory_usage'] = outputs['memory_limit'] = tf.no_op()

        with tf.device(device):

            #training part
            with tf.variable_scope('train'):

                #create the op to execute when done
                outputs['done'] = self._done(cluster)

                inputs, input_seq_length, targets, target_seq_length, num_steps\
                    = self._data(chief_ps)

                outputs['num_steps'] \
                    = num_steps*int(self.conf['num_epochs'])

                #create a check if training should continue
                outputs['should_stop'] = tf.logical_or(
                    tf.greater_equal(
                        outputs['global_step'],
                        outputs['num_steps']),
                    should_terminate)

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
                outputs['learning_rate'] = (tf.train.exponential_decay(
                    learning_rate=float(self.conf['initial_learning_rate']),
                    global_step=outputs['global_step'],
                    decay_steps=outputs['num_steps'],
                    decay_rate=float(self.conf['learning_rate_decay']))
                                            * learning_rate_fact)

                #compute the loss
                outputs['loss'] = loss_functions.factory(
                    self.conf['loss'])(
                        targets,
                        logits,
                        logit_seq_length,
                        target_seq_length)

                aditional_loss = self.aditional_loss()
                if aditional_loss is not None:
                    outputs['loss'] += aditional_loss

                outputs['update_op'] = self._update(
                    loss=outputs['loss'],
                    global_step=outputs['global_step'],
                    learning_rate=outputs['learning_rate'],
                    cluster=cluster)

            if self.evaluatorconf.get('evaluator', 'evaluator') != 'None':

                #validation part
                with tf.variable_scope('validate'):

                    #create a variable to hold the validation loss
                    outputs['validation_loss'] = tf.get_variable(
                        name='validation_loss',
                        shape=[],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0),
                        trainable=False)

                    #create a variable to save the last step where the model
                    #was validated
                    validated_step = tf.get_variable(
                        name='validated_step',
                        shape=[],
                        dtype=tf.int32,
                        initializer=tf.constant_initializer(
                            -int(self.conf['valid_frequency'])),
                        trainable=False)

                    #a check if validation is due
                    outputs['should_validate'] = tf.greater_equal(
                        outputs['global_step'] - validated_step,
                        int(self.conf['valid_frequency']))

                    val_batch_loss, outputs['valbatches'] = self._validate()

                    outputs['update_loss'] \
                        = outputs['validation_loss'].assign(
                            outputs['validation_loss'] +
                            val_batch_loss/outputs['valbatches']
                        ).op

                    #update the learning rate factor
                    outputs['half_lr'] = learning_rate_fact.assign(
                        learning_rate_fact/2).op

                    #create an operation to updated the validated step
                    outputs['update_validated_step'] \
                        = validated_step.assign(
                            outputs['global_step']).op

                    #variable to hold the best validation loss so far
                    outputs['best_validation'] = tf.get_variable(
                        name='best_validation',
                        shape=[],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(1.79e+308),
                        trainable=False)

                    #op to update the best velidation loss
                    outputs['update_best'] \
                        = outputs['best_validation'].assign(
                            outputs['validation_loss']).op

                    #a variable that holds the amount of workers at the
                    #validation point
                    waiting_workers = tf.get_variable(
                        name='waiting_workers',
                        shape=[],
                        dtype=tf.int32,
                        initializer=tf.constant_initializer(0),
                        trainable=False)

                    #an operation to signal a waiting worker
                    outputs['waiting'] = waiting_workers.assign_add(1).op

                    #an operation to set the waiting workers to zero
                    outputs['reset_waiting'] = waiting_workers.initializer

                    #an operation to check if all workers are waiting
                    if 'local' in cluster.as_dict():
                        outputs['all_waiting'] = tf.constant(True)
                    else:
                        outputs['all_waiting'] = tf.equal(
                            waiting_workers,
                            len(cluster.as_dict()['worker'])-1)

                    tf.summary.scalar('validation loss',
                                      outputs['validation_loss'])
            else:
                outputs['update_loss'] = None

            tf.summary.scalar('learning rate', outputs['learning_rate'])

            #create a histogram for all trainable parameters
            for param in tf.trainable_variables():
                tf.summary.histogram(param.name, param)

        return outputs

    def _data(self, chief_ps):
        '''
        create the input pipeline

        args:
            -chief_ps: the chief parameter server device

        returns:
            - the inputs
            - the input sequence lengths
            - the targets
            - the target sequence lengths
            - the number of steps in an epoch
        '''

        #get the database configurations
        input_names = self.model.conf.get('io', 'inputs').split(' ')
        if input_names == ['']:
            input_names = []
        input_sections = [self.conf[i].split(' ') for i in input_names]
        input_dataconfs = []
        for sectionset in input_sections:
            input_dataconfs.append([])
            for section in sectionset:
                input_dataconfs[-1].append(dict(self.dataconf.items(section)))

        output_names = self.conf['targets'].split(' ')
        if output_names == ['']:
            output_names = []
        target_sections = [self.conf[o].split(' ') for o in output_names]
        target_dataconfs = []
        for sectionset in target_sections:
            target_dataconfs.append([])
            for section in sectionset:
                target_dataconfs[-1].append(dict(self.dataconf.items(section)))

        #check if running in distributed model
        if chief_ps is None:

            #get the filenames
            data_queue_elements, _ = input_pipeline.get_filenames(
                input_dataconfs + target_dataconfs)

            #create the data queue and queue runners
            data_queue = tf.train.string_input_producer(
                string_tensor=data_queue_elements,
                shuffle=True,
                seed=None,
                capacity=int(self.conf['batch_size'])*2,
                shared_name='data_queue')

        else:
            with tf.device(chief_ps):

                #get the data queue
                data_queue = tf.FIFOQueue(
                    capacity=int(self.conf['batch_size'])*2,
                    shared_name='data_queue',
                    name='data_queue',
                    dtypes=[tf.string],
                    shapes=[[]])

        #create the input pipeline
        data, seq_length, num_steps = input_pipeline.input_pipeline(
            data_queue=data_queue,
            batch_size=int(self.conf['batch_size']),
            numbuckets=int(self.conf['numbuckets']),
            dataconfs=input_dataconfs + target_dataconfs,
            variable_batch_size=(
                self.conf['variable_batch_size'] == 'True')
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

        return inputs, input_seq_length, targets, target_seq_length, num_steps

    def _done(self, cluster):
        '''
        create the op to run when finished

        args:
            cluster: the tf cluster

        returns: the done op
        '''

        if 'local' in cluster.as_dict():
            done = tf.no_op()
        else:
            #get the done queues
            num_servers = len(cluster.as_dict()['ps'])
            num_replicas = len(cluster.as_dict()['worker'])
            done_ops = []
            for i in range(num_servers):
                with tf.device('job:ps/task:%d' % i):
                    done_queue = tf.FIFOQueue(
                        capacity=num_replicas,
                        dtypes=[tf.bool],
                        shapes=[[]],
                        shared_name='done_queue%d' % i,
                        name='done_queue%d' % i
                    )

                    done_ops.append(done_queue.enqueue(True))

            done = tf.group(*done_ops)

        return done

    def _validate(self):
        '''
        get the validation loss

        returns:
            - the validation loss for a batch
            - the number of validation batches
        '''

        #create the evaluator
        evaltype = self.evaluatorconf.get('evaluator', 'evaluator')
        if evaltype != 'None':
            evaluator = evaluator_factory.factory(evaltype)(
                conf=self.evaluatorconf,
                dataconf=self.dataconf,
                model=self.model
            )

        #compute the loss
        val_batch_loss, valbatches = evaluator.evaluate()

        return val_batch_loss, valbatches

    def _device(self, cluster):
        '''
        get the device

        args:
            cluster: a tf cluster

        returns:
            - the device specification
            - the chief paramater server device
        '''

        if 'local' in cluster.as_dict():
            device = tf.DeviceSpec(job='local')
            chief_ps = None
        else:
            #distributed training
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

        return device, chief_ps

    def _update(self, loss, global_step, learning_rate, cluster):
        '''
        create the op to update the model

        args:
            loss: the loss to minimize
            global_step: the gloabal step variable
            learning_rate: the learning rate
            cluster: the tf cluster

        returns: the update op
        '''

        #create the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        #create an optimizer that aggregates gradients
        if int(self.conf['numbatches_to_aggregate']) > 0:
            if 'local' in cluster.as_dict():
                num_workers = 1
            else:
                num_workers = len(cluster.as_dict()['worker'])

            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=int(
                    self.conf['numbatches_to_aggregate']),
                total_num_replicas=num_workers)

        #a variable to store the loss
        training_loss = tf.get_variable(
            name='loss',
            shape=[],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
            trainable=False)

        tf.summary.scalar('training loss', training_loss)

        #an op to update the loss
        update_loss = training_loss.assign(loss).op

        #get the list of trainable variables
        trainable = tf.trainable_variables()

        #get the list of variables to be removed from the trainable
        #variables
        untrainable = tf.get_collection('untrainable')

        #remove the variables
        trainable = [var for var in trainable
                     if var not in untrainable]

        #compute the gradients
        grads_and_vars = optimizer.compute_gradients(
            loss=loss,
            var_list=trainable)

        with tf.variable_scope('clip'):
            #clip the gradients
            grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var)
                              for grad, var in grads_and_vars]


        #opperation to apply the gradients
        apply_gradients_op = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars,
            global_step=global_step,
            name='apply_gradients')

        #all remaining operations with the UPDATE_OPS GraphKeys
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #create an operation to update the gradients, the batch_loss
        #and do all other update ops
        update_op = tf.group(
            *([apply_gradients_op, update_loss] + update_ops),
            name='update')

        return update_op

    def train(self):
        '''train the model'''

        #look for the master if distributed training is done
        master = self.server.target

        #start the session and standart servises
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        #number of times validation performance was worse
        num_tries = 0

        #check if this is the chief worker
        is_chief = self.task_index == 0

        #create the graph
        graph = tf.Graph()
        with graph.as_default():
            outputs = self._create_graph()
            scaffold = tf.train.Scaffold()

        #create a hook for saving the final model
        save_hook = hooks.SaveAtEnd(
            os.path.join(self.expdir, 'model', 'network.ckpt'),
            self.model.variables)

        #create a hook for saving and restoring the validated model
        validation_hook = hooks.ValidationSaveHook(
            os.path.join(self.expdir, 'logdir', 'validated.ckpt'),
            self.model)

        #create the summary hook
        summary_hook = hooks.SummaryHook(os.path.join(self.expdir, 'logdir'))

        with graph.as_default():
            with tf.train.MonitoredTrainingSession(
                master=master,
                is_chief=is_chief,
                checkpoint_dir=os.path.join(self.expdir, 'logdir'),
                scaffold=scaffold,
                hooks=[hooks.StopHook(outputs['done'])] + self.hooks(outputs),
                chief_only_hooks=[save_hook, validation_hook, summary_hook] \
                    + self.chief_only_hooks(outputs),
                config=config) as sess:

                #start the training loop
                #pylint: disable=E1101
                while not (sess.should_stop() or
                           outputs['should_stop'].eval(session=sess)):

                    #check if validation is due
                    if (outputs['update_loss'] is not None
                            and outputs['should_validate'].eval(session=sess)):
                        if is_chief:
                            print ('WORKER %d: validating model'
                                   % self.task_index)

                            #get the previous validation loss
                            prev_val_loss = outputs['best_validation'].eval(
                                session=sess)

                            #reset the validation loss
                            outputs['validation_loss'].initializer.run(
                                session=sess)

                            #compute the validation loss
                            for _ in range(outputs['valbatches']):
                                outputs['update_loss'].run(session=sess)

                            #get the current validation loss
                            validation_loss = outputs['validation_loss'].eval(
                                session=sess)

                            print ('WORKER %d: validation loss: %f' %
                                   (self.task_index, validation_loss))

                            #check if the validation loss is better
                            if validation_loss >= prev_val_loss:

                                print ('WORKER %d: validation loss is worse' %
                                       self.task_index)

                                #check how many times validation performance was
                                #worse
                                if self.conf['num_tries'] != 'None':
                                    if num_tries == int(self.conf['num_tries']):
                                        validation_hook.restore()
                                        print ('WORKER %d: terminating training'
                                               % self.task_index)
                                        outputs['terminate'].run(session=sess)
                                        break

                                num_tries += 1

                                if self.conf['go_back'] == 'True':

                                    #wait untill all workers are at validation
                                    #point
                                    while not outputs['all_waiting'].eval(
                                            session=sess):
                                        time.sleep(1)
                                    outputs['reset_waiting'].run(session=sess)

                                    print ('WORKER %d: loading previous model'
                                           % self.task_index)

                                    #load the previous model
                                    validation_hook.restore()
                                else:
                                    outputs['update_validated_step'].run(
                                        session=sess)


                                if self.conf['valid_adapt'] == 'True':
                                    print ('WORKER %d: halving learning rate'
                                           % self.task_index)
                                    outputs['half_lr'].run(session=sess)
                                    validation_hook.save()

                            else:
                                if self.conf['reset_tries'] == 'True':
                                    num_tries = 0

                                #set the validated step
                                outputs['update_validated_step'].run(
                                    session=sess)
                                outputs['update_best'].run(session=sess)
                                outputs['reset_waiting'].run(session=sess)

                                #store the validated model
                                validation_hook.save()

                        else:
                            if (self.conf['go_back'] == 'True'
                                    and self.update_loss is not None):
                                outputs['waiting'].run(session=sess)
                                while (
                                        outputs['should_validate'].eval(
                                            session=sess)
                                        and not
                                        outputs['should_stop'].eval(
                                            session=sess)):
                                    time.sleep(1)

                                if outputs['should_stop'].eval(session=sess):
                                    break

                    #start time
                    start = time.time()

                    #update the model
                    _, loss, lr, global_step, memory, limit = \
                        sess.run(
                            fetches=[outputs['update_op'],
                                     outputs['loss'],
                                     outputs['learning_rate'],
                                     outputs['global_step'],
                                     outputs['memory_usage'],
                                     outputs['memory_limit']])

                    if memory is not None:
                        memory_line = '\n\t peak memory usage: %d/%d MB' % (
                            memory/1e6,
                            limit/1e6
                        )
                    else:
                        memory_line = ''

                    print(('WORKER %d: step %d/%d loss: %f, learning rate: %f '
                           '\n\t time elapsed: %f sec%s')
                          %(self.task_index,
                            global_step,
                            outputs['num_steps'],
                            loss, lr, time.time()-start,
                            memory_line))

        #store the model file
        modelfile = os.path.join(self.expdir, 'model', 'model.pkl')
        with open(modelfile, 'wb') as fid:
            pickle.dump(self.model, fid)

    @abstractmethod
    def chief_only_hooks(self, outputs):
        '''add hooks only for the chief worker

        Args:
            outputs: the outputs generated by the create graph method

        Returns:
            a list of hooks'''

    @abstractmethod
    def hooks(self, outputs):
        '''add hooks for the session

        Args:
            outputs: the outputs generated by the create graph method

        Returns:
            a list of hooks
        '''

    @abstractmethod
    def aditional_loss(self):
        '''add an aditional loss

        returns:
            the aditional loss or None'''

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
                    capacity=int(conf['batch_size'])*2,
                    shared_name='data_queue')

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
                is_chief=False,
                scaffold=self.scaffold) as sess:

                self.wait_op.run(session=sess)

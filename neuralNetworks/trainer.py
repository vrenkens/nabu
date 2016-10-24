'''@file trainer.py
neural network trainer environment'''

import tensorflow as tf

class Trainer(object):
    '''Class for the training environment for a neural net graph'''

    def __init__(self, classifier, input_dim, init_learning_rate, learning_rate_decay, num_steps, numframes_per_batch):
        '''
        NnetTrainer constructor, creates the training graph

        Args:
            classifier: the neural net classifier that will be trained
            input_dim: the input dimension to the nnnetgraph
            init_learning_rate: the initial learning rate
            learning_rate_decay: the parameter for exponential learning rate decay
            num_steps: the total number of steps that will be taken
            numframes_per_batch: determines how many frames are processed at a time to limit memory usage
        '''

        self.numframes_per_batch = numframes_per_batch

        #create the graph
        self.graph = tf.Graph()

        #define the placeholders in the graph
        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(tf.float32, shape=[None, input_dim], name='inputs')

            #reference labels
            self.targets = tf.placeholder(tf.float32, shape=[None, classifier.output_dim], name='targets')

            #input for the total number of frames that are used in the batch
            self.num_frames = tf.placeholder(tf.float32, shape=[], name='num_frames')

            #compute the training outputs of the nnetgraph
            trainlogits, self.modelsaver, self.control_ops = classifier(self.inputs, is_training=True, reuse=False, scope='Classifier')

            #compute the validation output of the nnetgraph
            logits, _, _ = classifier(self.inputs, is_training=False, reuse=True, scope='Classifier')

            #get a list of trainable variables in the decoder graph
            params = tf.trainable_variables()

            #add the variables and operations to the graph that are used for training

            #total number of steps
            nsteps = tf.constant(num_steps, dtype=tf.int32, name='num_steps')

            #the total loss of the entire batch
            batch_loss = tf.get_variable('batch_loss', [], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)

            with tf.variable_scope('train_variables'):

                #the amount of steps already taken
                self.global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

                #a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
                learning_rate_fact = tf.get_variable('learning_rate_fact', [], initializer=tf.constant_initializer(1.0), trainable=False)

                #compute the learning rate with exponential decay and scale with the learning rate factor
                learning_rate = tf.train.exponential_decay(init_learning_rate, self.global_step, nsteps, learning_rate_decay) * learning_rate_fact

                #create the optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate)

            #for every parameter create a variable that holds its gradients
            with tf.variable_scope('gradients'):
                grads = [tf.get_variable(param.op.name, param.get_shape().as_list(), initializer=tf.constant_initializer(0), trainable=False) for param in params]

            with tf.name_scope('train'):
                #compute the training loss
                loss = tf.reduce_sum(self.compute_loss(self.targets, trainlogits))

                #operation to half the learning rate
                self.halve_learningrate_op = learning_rate_fact.assign(learning_rate_fact/2).op

                #create an operation to initialise the gradients
                self.init_grads = tf.initialize_variables(grads)

                #the operation to initialise the batch loss
                self.init_loss = batch_loss.initializer #pylint: disable=E1101

                #compute the gradients of the batch
                batchgrads = tf.gradients(loss, params)

                #create an operation to update the batch loss
                self.update_loss = batch_loss.assign(batch_loss+loss).op

                #create an operation to update the gradients, the batch_loss and do all other update ops
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.update_gradients_op = tf.group(*([grads[p].assign_add(batchgrads[p]) for p in range(len(grads)) if batchgrads[p] is not None] + [self.update_loss] + update_ops), name='update_gradients')

                #create an operation to apply the gradients
                meangrads = [tf.div(grad, self.num_frames, name=grad.op.name) for grad in grads]
                self.apply_gradients_op = optimizer.apply_gradients([(meangrads[p], params[p]) for p in range(len(meangrads))], global_step=self.global_step, name='apply_gradients')

            with tf.name_scope('valid'):
                #compute the validation loss
                valid_loss = tf.reduce_sum(self.compute_loss(self.targets, logits))

                #operation to update the validation loss
                self.update_valid_loss = batch_loss.assign(batch_loss+valid_loss).op

            #operation to compute the average loss in the batch
            self.average_loss = batch_loss/self.num_frames

            # add an operation to initialise all the variables in the graph
            self.init_op = tf.initialize_all_variables()

            #saver for the training variables
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='train_variables'))

            #create the summaries for visualisation
            self.summary = tf.merge_summary([tf.histogram_summary(val.name, val) for val in params+meangrads] + [tf.scalar_summary('loss', self.average_loss)])


        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

        #start without visualisation
        self.summarywriter = None

    def compute_loss(self, targets, logits): #pylint: disable=R0201
        '''
        Creates the operation to compute the cross-enthropy loss for every input frame (if you want to have a different loss function, overwrite this method)

        Args:
            targets: a NxO tensor containing the reference targets where N is the number of frames and O is the neural net output dimension
            logits: a NxO tensor containing the neural network output logits where N is the number of frames and O is the neural net output dimension

        Returns:
            an N-dimensional tensor containing the losses for all the input frames where N is the number of frames
        '''

        return tf.nn.softmax_cross_entropy_with_logits(logits, targets, name='loss')

    def initialize(self):
        '''Initialize all the variables in the graph'''

        self.init_op.run() #pylint: disable=E1101

    def start_visualization(self, logdir):
        '''
        open a summarywriter for visualisation and add the graph

        Args:
            logdir: directory where the summaries will be written
        '''

        self.summarywriter = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)

    def update(self, inputs, targets):
        '''
        update the neural model with a batch or training data

        Args:
            inputs: the inputs to the neural net, this should be a NxF numpy array where N is the number of frames in the batch and F is the feature dimension
            targets: the one-hot encoded targets for neural nnet, this should be an NxO matrix where O is the output dimension of the neural net

        Returns:
            the loss at this step
        '''

        #if numframes_per_batch is not set just process the entire batch
        if self.numframes_per_batch == -1 or self.numframes_per_batch > inputs.shape[0]:
            numframes_per_batch = inputs.shape[0]
        else:
            numframes_per_batch = self.numframes_per_batch

        #feed in the batches one by one and accumulate the gradients and loss
        for k in range(int(inputs.shape[0]/numframes_per_batch) + int(inputs.shape[0]%numframes_per_batch > 0)):
            batch_inputs = inputs[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
            batch_targets = targets[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
            self.update_gradients_op.run(feed_dict={self.inputs:batch_inputs, self.targets:batch_targets}) #pylint: disable=E1101

        #apply the accumulated gradients to update the model parameters and evaluate the loss
        if self.summarywriter is not None:
            [loss, summary, _] = tf.get_default_session().run([self.average_loss, self.summary, self.apply_gradients_op], feed_dict={self.num_frames:inputs.shape[0]})
            self.summarywriter.add_summary(summary, global_step=self.global_step.eval()) #pylint: disable=E1101
        else:
            [loss, _] = tf.get_default_session().run([self.average_loss, self.apply_gradients_op], feed_dict={self.num_frames:inputs.shape[0]})


        #reinitialize the gradients and the loss
        self.init_grads.run() #pylint: disable=E1101
        self.init_loss.run()

        return loss

    def evaluate(self, inputs, targets):
        '''
        Evaluate the performance of the neural net

        Args:
            inputs: the inputs to the neural net, this should be a NxF numpy array where N is the number of frames in the batch and F is the feature dimension
            targets: the one-hot encoded targets for neural nnet, this should be an NxO matrix where O is the output dimension of the neural net

        Returns:
            the loss of the batch
        '''

        if inputs is None or targets is None:
            return None

        #if numframes_per_batch is not set just process the entire batch
        if self.numframes_per_batch == -1 or self.numframes_per_batch > inputs.shape[0]:
            numframes_per_batch = inputs.shape[0]
        else:
            numframes_per_batch = self.numframes_per_batch

        #feed in the batches one by one and accumulate the loss
        for k in range(int(inputs.shape[0]/numframes_per_batch) + int(inputs.shape[0]%numframes_per_batch > 0)):
            batch_inputs = inputs[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
            batch_targets = targets[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
            self.update_valid_loss.run(feed_dict={self.inputs:batch_inputs, self.targets:batch_targets})

        #get the loss
        loss = self.average_loss.eval(feed_dict={self.num_frames:inputs.shape[0]})

        #reinitialize the loss
        self.init_loss.run()

        return loss

    def halve_learning_rate(self):
        '''halve the learning rate'''

        self.halve_learningrate_op.run()

    def save_model(self, filename):
        '''
        Save the model

        Args:
            filename: path to the model file
        '''
        self.modelsaver.save(tf.get_default_session(), filename)

    def restore_model(self, filename):
        '''
        Load the model

        Args:
            filename: path where the model will be saved
        '''
        self.modelsaver.restore(tf.get_default_session(), filename)

    def save_trainer(self, filename):
        '''
        Save the training progress (including the model)

        Args:
            filename: path where the model will be saved
        '''

        self.modelsaver.save(tf.get_default_session(), filename)
        self.saver.save(tf.get_default_session(), filename + '_trainvars')

    def restore_trainer(self, filename):
        '''
        Load the training progress (including the model)

        Args:
            filename: path where the model will be saved
        '''

        self.modelsaver.restore(tf.get_default_session(), filename)
        self.saver.restore(tf.get_default_session(), filename + '_trainvars')

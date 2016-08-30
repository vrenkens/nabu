import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
import itertools

import nnetlayer

#this an abstrace class defining a neural net that is used for decoding
class NnetGraph(object):
	__metaclass__ = ABCMeta
	
	def __init__(self, name, *args, **kwargs):
		
		self.name = name
		
		if len(args) + len(kwargs) < len(self.fieldnames):
			raise TypeError('%s() expects %d arguments (%d given)' %(type(self).__name__, len(self.fieldnames), len(args) + len(kwargs)))
			
		for a in range(len(args)):
			exec('self.%s = args[a]' % self.fieldnames[a])
			
		for a in kwargs:
			if a not in self.fieldnames:
				raise TypeError('%s is an invalid keyword argument for %s()' % (a, type(self).__name__))
			
			exec('self.%s = kwargs[a]' % (a))
	
	#this method extends the graph with the decoding graph, this method should define the attributes: inputs, outputs, logits and saver. 
	#	inputs: the placeholder for the inputs
	#	outputs: this is outputed if the network is called
	#	logits:	the is used when computing the cross enthropy when training the network
	#	saver: the saver for the model parameters
	#	graph: the graph that is extended
	#	*args, **kwargs: the other arguments that are needed for the method
	@abstractmethod
	def extendGraph(self):
		pass			
		
	#this property should return the a list of strings containing the fielnames of the __init__ function
	@abstractproperty
	def fieldnames(self):
		pass	
	
	#this property should return the inputs placeholder
	@property
	def inputs(self):
		return self._inputs
		
	#this property should return the outputs of the graph
	@property
	def outputs(self):
		return self._outputs
		
	#this property should return the logits used for training
	@property
	def logits(self):
		return self._logits
		
	#this property should return the saver of the model parameters
	@property
	def saver(self):
		return self._saver
		
	@inputs.setter
	def inputs(self, inputs):
		self._inputs = inputs
		
	@outputs.setter
	def outputs(self, outputs):
		self._outputs = outputs
		
	@logits.setter
	def logits(self, logits):
		self._logits = logits
		
	@saver.setter
	def saver(self, saver):
		self._saver = saver
		
	
		
#This class is a graph for feedforward fully connected neural nets
# it is initialised as folows: DNN(name, input_dim, output_dim, num_hidden_layers, num_hidden_units, transfername, l2_norm, dropout)'
#	name: name of the DNN
# 	input_dim: the input dimension
#	output_dim: the output dimension
#	num_hidden_layers: number of hiden layers in the DNN
#	layer_wise_init: Boolean that is true if layerwhise initialisation should be done
#	num_hidden_units: number of hidden units in every layer
#	transfername: name of the transfer function that is used
#	l2_norm: boolean that determines of l2_normalisation is used after every layer
#	dropout: the chance that a hidden unit is propagated to the next layer
class DNN(NnetGraph):
	
	#extends the graph with the DNN
	def extendGraph(self):
			
		with tf.variable_scope(self.name):
			
			#define the input data
			self.inputs = tf.placeholder(tf.float32, shape = [None, self.input_dim], name = 'inputs')
		
			#placeholder to set the state prior
			self.prior = tf.placeholder(tf.float32, shape = [self.output_dim], name = 'priorGate')
		
			#variable that holds the state prior
			stateprior = tf.get_variable('prior', self.output_dim, initializer=tf.constant_initializer(0), trainable=False)
		
			#variable that holds the state prior
			initialisedlayers = tf.get_variable('initialisedlayers', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
		
			#operation to increment the number of layers
			self.addLayerOp = initialisedlayers.assign_add(1).op
		
			#operation to set the state prior
			self.setPriorOp = stateprior.assign(self.prior).op
		
			#create the layers
			layers = [None]*(self.num_hidden_layers+1)
		 	
		 	#input layer
		 	layers[0] = nnetlayer.FFLayer(self.input_dim, self.num_hidden_units, 1/np.sqrt(self.input_dim), 'layer0', self.transfername, self.l2_norm, self.dropout)
		 	
		 	#hidden layers
		 	for k in range(1,len(layers)-1):
		 		layers[k] = nnetlayer.FFLayer(self.num_hidden_units, self.num_hidden_units, 1/np.sqrt(self.num_hidden_units), 'layer' + str(k), self.transfername, self.l2_norm, self.dropout)
		 		
	 		#output layer
	 		layers[-1] = nnetlayer.FFLayer(self.num_hidden_units, self.output_dim, 0, 'layer' + str(len(layers)-1))
	 		
	 		#operation to initialise the final layer
	 		self.initLastLayerOp = tf.initialize_variables([layers[-1].weights, layers[-1].biases])
	 		
	 		#compute the activations at each hidden layer
	 		activations = [None]*(len(layers)-1)
			activations[0] = layers[0](self.inputs)
			for l in range(1,len(activations)):
				activations[l] = layers[l](activations[l-1])
	 		
	 		if self.layer_wise_init:
				#compute the logits by selecting the activations at the layer that has last been added to the network, this is used for layer by layer initialisation
				self.logits = layers[-1](tf.case([(tf.equal(initialisedlayers, tf.constant(l)), callableTensor(activations[l])) for l in range(len(activations))], callableTensor(activations[-1]),name = 'layerSelector'))
			else:
				self.logits = layers[-1](activations[-1])
	 				
			#define the output 
			self.outputs = tf.nn.softmax(self.logits)/stateprior
		
			#normalise the output
			self.outputs = self.outputs/tf.reduce_sum(self.outputs,1, keep_dims=True)
		
			#create a saver 
			self.saver = tf.train.Saver()
	
	#this function sets the prior in the graph
	#	prior: the state prior probabilities
	def setPrior(self, prior):
		self.setPriorOp.run(feed_dict={self.prior:prior})
		
	#this function adds a layer to the network
	def addLayer(self):
		#reinitialise the final layer
		self.initLastLayerOp.run()
		
		#increment the number of layers
		self.addLayerOp.run()
		
	@property
	def fieldnames(self):
		return ['input_dim', 'output_dim', 'num_hidden_layers',  'layer_wise_init', 'num_hidden_units', 'transfername', 'l2_norm', 'dropout']

#this objects creates a decoding environment for a neural net graph
class NnetDecoder(object):
	def __init__(self, nnetGraph):

		self.graph = tf.Graph()
		self.nnetGraph = nnetGraph
		
		with self.graph.as_default():
		
			#create the decoding graph
			self.nnetGraph.extendGraph()
	
		#specify that the graph can no longer be modified after this point
		self.graph.finalize()
	
	#decode using the graph
	#	inputs: the inputs to the graph as a N*F numpy array where N is the number of frames and F is the input feature dimension
	#	returns: the output of the neural net
	def __call__(self, inputs):
		return self.nnetGraph.outputs.eval(feed_dict = {self.nnetGraph.inputs:inputs})
	
	#saves the neural net
	#	filename: location where the neural net is saved
	def save(self, filename):
		self.nnetGraph.saver.save(tf.get_default_session(), filename)
	
	#load a saved neural net
	#	filename: location where the neural net is saved
	def restore(self, filename):
		self.nnetGraph.saver.restore(tf.get_default_session(), filename)
	
 					
#this objects creates a training environment for a neural net graph
class NnetTrainer(object):

	#create the trainer object
	#	nnetGraph: the NnetGraph object
	#	init_learning_rate: the initial learning rate
	#	learning_rate_decay: the parameter for exponential learning rate decay
	#	num_steps: the total number of steps that will be taken
	#	numframes_per_batch: determines how many frames are processed at a time to limit memory usage
	def __init__(self, nnetGraph, init_learning_rate, learning_rate_decay, num_steps, numframes_per_batch):
	
		self.numframes_per_batch = numframes_per_batch
		self.nnetGraph = nnetGraph
	
		#create the graph
		self.graph = tf.Graph()
		
		#define the placeholders in the graph
		with self.graph.as_default():
			
			#create the decoding graph
			self.nnetGraph.extendGraph()
			
			#reference labels
			self.targets = tf.placeholder(tf.float32, shape = [None, self.nnetGraph.logits.get_shape().as_list()[1]], name = 'targets')
			
			#input for the total number of frames that are used in the batch
			self.num_frames = tf.placeholder(tf.float32, shape = [], name = 'num_frames')
			
			#get a list of trainable variables in the decoder graph
			params = tf.trainable_variables()

			#add the variables and operations to the graph that are used for training
			
			#compute the loss
			self.loss = tf.reduce_sum(self.computeLoss(self.targets, self.nnetGraph.logits))
			
			#total number of steps
			Nsteps = tf.constant(num_steps, dtype = tf.int32, name = 'num_steps')
			
			#the total loss of the entire batch
			batch_loss = tf.get_variable('batch_loss', [], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
			
			with tf.variable_scope('train_variables'):	

				#the amount of steps already taken
				self.global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False) 
	
				#a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
				learning_rate_fact = tf.get_variable('learning_rate_fact', [], initializer=tf.constant_initializer(1.0), trainable=False)
				
				#compute the learning rate with exponential decay and scale with the learning rate factor
				learning_rate = tf.train.exponential_decay(init_learning_rate, self.global_step, Nsteps, learning_rate_decay) * learning_rate_fact
				
				#create the optimizer
				optimizer = tf.train.AdamOptimizer(learning_rate)
			
			#for every parameter create a variable that holds its gradients
			with tf.variable_scope('gradients'):
				grads = [tf.get_variable(param.op.name, param.get_shape().as_list(), initializer=tf.constant_initializer(0), trainable=False) for param in params]
				
			with tf.name_scope('train'):
				#operation to half the learning rate
				self.halveLearningRateOp = learning_rate_fact.assign(learning_rate_fact/2).op
				
				#create an operation to initialise the gradients
				self.initgrads = tf.initialize_variables(grads)
				
				#the operation to initialise the batch loss
				self.initloss = batch_loss.initializer
				
				#compute the gradients of the batch
				batchgrads = tf.gradients(self.loss, params)
				
				#create an operation to update the batch loss
				self.updateLoss = batch_loss.assign_add(self.loss).op
				
				#create an operation to update the gradients and the batch_loss
				self.updateGradientsOp = tf.group(*([grads[p].assign_add(batchgrads[p]) for p in range(len(grads)) if batchgrads[p] is not None] + [self.updateLoss]), name='update_gradients')
				
				#create an operation to apply the gradients
				self.applyGradientsOp = optimizer.apply_gradients([(grads[p]/self.num_frames, params[p]) for p in range(len(grads))], global_step=self.global_step, name='apply_gradients')
				
				# add an operation to initialise all the variables in the graph
				self.initop = tf.initialize_all_variables()
			
				#operation to compute the average loss in the batch
				self.average_loss = batch_loss/self.num_frames
			
			#saver for the training variables
			self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='train_variables'))
			
			#create the summaries for visualisation
			self.summary = tf.merge_summary([tf.histogram_summary(val.name, val) for val in params+grads] + [tf.scalar_summary('loss', batch_loss)])
			
			
		#specify that the graph can no longer be modified after this point
		self.graph.finalize()
		
		#start without visualisation
		self.summarywriter = None
		
	#this method creates the operation to compute the cross-enthropy loss for every input frame (if you want to have a different loss function, overwrite this method)
	def computeLoss(self, targets, logits):
		return tf.nn.softmax_cross_entropy_with_logits(logits, targets, name='loss')
		
	#this method initialises all the variables in the neural net
	def initialize(self):
		self.initop.run()
		
	#open a summarywriter for visualisation and add the graph
	#	logdir: location where the summaries will be written
	def startVisualization(self, logdir):
		self.summarywriter = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)
	
	#this method updates the neural net model
	#	inputs: the inputs to the neural net, this should be a NxF numpy array where N is the number of frames in the batch and F is the feature dimension
	#	targets: the one-hot encoded targets for neural nnet, this should be an NxP matrix where P is the output dimension of the neural net
	#	returns: the loss at this step
	def update(self, inputs, targets):
		
		#if numframes_per_batch is not set just process the entire batch
		if self.numframes_per_batch==-1 or self.numframes_per_batch>inputs.shape[0]:
			numframes_per_batch = inputs.shape[0]
		else:
			numframes_per_batch = self.numframes_per_batch
				
		#feed in the batches one by one and accumulate the gradients and loss
		for k in range(int(inputs.shape[0]/numframes_per_batch) + int(inputs.shape[0]%numframes_per_batch > 0)):
			batchInputs = inputs[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
			batchTargets = targets[k*numframes_per_batch:min((k+1)*numframes_per_batch, inputs.shape[0]), :]
			self.updateGradientsOp.run(feed_dict = {self.nnetGraph.inputs:batchInputs, self.targets:batchTargets})
			
		#apply the accumulated gradients to update the model parameters
		self.applyGradientsOp.run(feed_dict = {self.num_frames:inputs.shape[0]})
		
		#get the loss at this step
		loss = self.average_loss.eval(feed_dict = {self.num_frames:inputs.shape[0]})
		
		#if visualization has started add the summary
		if self.summarywriter is not None:
			self.summarywriter.add_summary(self.summary.eval(), global_step=self.global_step.eval())
		
		#reinitialize the gradients and the loss
		self.initgrads.run()
		self.initloss.run()
		
		return loss
		

	#this method is used to evaluate the performance of the neural net
	#	inputs: the inputs to the neural net, this should be a NxF numpy array where N is the number of frames in the batch and F is the feature dimension
	#	targets: the one-hot encoded targets for neural nnet, this should be an NxP matrix where P is the output dimension of the neural net
	#	returns: the loss of the batch
	def evaluate(self, inputs, targets):
		
		if inputs is None or targets is None:
			return None
	
		#if numframes_per_batch is not set just process the entire batch
		if self.numframes_per_batch==-1 or self.numframes_per_batch>inputs.shape[0]:
			numframes_per_batch = inputs.shape[0]
		else:
			numframes_per_batch = self.numframes_per_batch
					
		#feed in the batches one by one and accumulate the loss
		for k in range(int(inputs.shape[0]/self.numframes_per_batch) + int(inputs.shape[0]%self.numframes_per_batch > 0)):
			batchInputs = inputs[k*self.numframes_per_batch:min((k+1)*self.numframes_per_batch, inputs.shape[0]), :]
			batchTargets = targets[k*self.numframes_per_batch:min((k+1)*self.numframes_per_batch, inputs.shape[0]), :]
			self.updateLoss.run(feed_dict = {self.nnetGraph.inputs:batchInputs, self.targets:batchTargets})
			
		#get the loss
		loss = self.average_loss.eval(feed_dict = {self.num_frames:inputs.shape[0]})
		
		#reinitialize the loss
		self.initloss.run()
		
		return loss
		
			
	#this method halves the learning rate
	def halve_learning_rate(self):
		self.halveLearningRateOp.run()
	
	#this method is used to save the model
	#	filename: filename of the model file
	def saveModel(self, filename):
		self.nnetGraph.saver.save(tf.get_default_session(), filename)
		
	#this method is used to load the model
	#	filename: filename of the model file
	def restoreModel(self, filename):
		self.nnetGraph.saver.restore(tf.get_default_session(), filename)
		
	#this method is used to save training progress (including the model)
	#	filename: filename of the model file
	def saveTrainer(self, filename):
		self.nnetGraph.saver.save(tf.get_default_session(), filename)
		self.saver.save(tf.get_default_session(), filename + '_trainvars')
		
	#this method is used to load training progress (including the model)
	#	filename: filename of the model file
	def restoreTrainer(self, filename):
		self.nnetGraph.saver.restore(tf.get_default_session(), filename)
		self.saver.restore(tf.get_default_session(), filename + '_trainvars')
		
class callableTensor:
	def __init__(self, tensor):
		self.tensor = tensor
	def __call__(self):
		return self.tensor
			


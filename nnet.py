import kaldi_io
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import gzip
import shutil
import os

#compute the accuracy of predicted labels
#	predictions: predicted labels
#	labels: reference alignments
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)))

#splice the utterance
#	utt: utterance to be spliced
#	context width: how many franes to the left and right should be concatenated
def splice(utt, context_width):
	#create spliced utterance holder
	utt_spliced = np.zeros(shape = [utt.shape[0],utt.shape[1]*(1+2*context_width)], dtype=np.float32)
	#middle part is just the uttarnce
	utt_spliced[:,context_width*utt.shape[1]:(context_width+1)*utt.shape[1]] = utt
	for i in range(context_width):
		#add left context
		utt_spliced[i+1:utt_spliced.shape[0], (context_width-i-1)*utt.shape[1]:(context_width-i)*utt.shape[1]] = utt[0:utt.shape[0]-i-1,:]
	 	#add right context	
		utt_spliced[0:utt_spliced.shape[0]-i-1, (context_width+i+1)*utt.shape[1]:(context_width+i+2)*utt.shape[1]] = utt[i+1:utt.shape[0],:]
	
	return utt_spliced
	
#apply mean and variance normalisation based on the previously computed statistics
#	utt: the utterance feature matrix
#	stats: the mean and variance statistics
def apply_cmvn(utt, stats):
	#compute mean
	mean = stats[0,0:stats.shape[1]-1]/stats[0,stats.shape[1]-1]
	#compute variance
	variance = stats[1,0:stats.shape[1]-1]/stats[0,stats.shape[1]-1] - np.square(mean)
	#return mean and variance normalised utterance
	return np.divide(np.subtract(utt, mean), np.sqrt(variance))

#create a batch of data
#	reader: feature reader
#	reader_cmvn: reader for mean and variane statistics
#	alignments: dictionary containing the state alignments
#	utt2spk: mapping from utterance to speaker
#	input dim: dimension of the input features
#	context width: number of left and right frames used for splicing
#	num_labels: number of output labels
#	batch_size: size of the batch to be created in number of utterances
def create_batch(reader, reader_cmvn, alignments, utt2spk, input_dim, context_width, num_labels, batch_size):
	#create empty batch for input features
	batch_data = np.empty([0,input_dim*(1+2*context_width)], dtype=np.float32)
	#create empty batch for labels
	batch_labels = np.empty([0,0], dtype=np.float32)
	#initialise number of utterance in the batch
	num_utt = 0
	while num_utt < batch_size:
		#read utterance
		(utt_id, utt_mat, _) = reader.read_next_utt()
		#check if utterance has an alignment
		if utt_id in alignments:
			#read cmvn stats
			stats = reader_cmvn.read_utt(utt2spk[utt_id])
			#apply cmvn
			utt_mat = apply_cmvn(utt_mat, stats)		
			#add the spliced utterance to batch			
			batch_data = np.append(batch_data, splice(utt_mat,context_width), axis=0)			
			#add labels to batch
			batch_labels = np.append(batch_labels, alignments[utt_id])
			#update number iof utterances in the batch
			num_utt = num_utt + 1
		else:
			print('WARNING no alignment for %s' % utt_id)
			
	#put labels in one hot encoding
	batch_labels = (np.arange(num_labels) == batch_labels[:,None]).astype(np.float32)
	
	return (batch_data, batch_labels)

class Nnet:
	#create nnet and define variables in the computational graph
	#	conf: nnet configuration
	def __init__(self, conf):
		self.conf = conf

	#propagate does the computations of a single layer 
	#	data: input data to the layer
	#	w: weight matrix
	#	b: bias vector
	#	dropout: dropout to be applied (should only be used in training)
	def propagate(self,data, w, b, dropout):

		#apply weights and biases
		data = tf.matmul(data, w) + b
		#apply non linearity, current opions are relu, sigmoid and hyperbolic tangent
		if self.conf['nonlin'] == 'relu':
			data = tf.maximum(float(self.conf['relu_leak'])*data,data)
		elif self.conf['nonlin'] == 'sigmoid':
			data = tf.nn.sigmoid(data)
		elif self.conf['nonlin'] == 'tanh':
			data = tf.nn.tanh(data)
		else:
			raise Exception('unknown nonlinearity')
		#apply l2 normalisation
		if self.conf['l2_norm'] == 'True':
			data = tf.nn.l2_normalize(data,1)*np.sqrt(float(self.conf['num_hidden_units']))		
		#apply dropout	
		if dropout<1:
			data = tf.nn.dropout(data, dropout)
	
		return data

	#model propagates the data through the entire neural net
	#	data: input data to the neural net
	#	num_layers: number of hidden layers that should be used
	#	dropout: dropout to be applied (should only be used in training)
	def model(self,data, weights, biases, dropout):	
		#propagate through the neural net
		for i in range(len(weights)):
			data = self.propagate(data, weights[i], biases[i], dropout)
		
		return data

	#creats the basic graph for the nnet
	#	num_layers: number of hidden layers in the neural net (> 0)
	#	returns the computational graph and neural net
	def create_graph(self, num_layers):
		nnet_dict = {}
		graph = tf.Graph()
		with graph.as_default():
			#input data
			nnet_dict['data_in'] = tf.placeholder(tf.float32, shape = [None, self.conf['input_dim']*(1+2*int(self.conf['context_width']))])
		
			#define weights, biases and their derivatives lists
			nnet_dict['weights'] = []
			nnet_dict['biases'] = []
		
			#input layer, initialise as random normal
			nnet_dict['weights'].append(tf.Variable(tf.random_normal([self.conf['input_dim']*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])], stddev=1/np.sqrt(self.conf['input_dim'])), name = 'Win'))
			nnet_dict['biases'].append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units'])], stddev=float(self.conf['biases_std'])), name = 'bin'))
		
			#hidden layers, initialise as random normal
			for i in range(num_layers-1):
				nnet_dict['weights'].append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])], stddev=1/np.sqrt(float(self.conf['num_hidden_units']))), name = 'W%d' % i))
				nnet_dict['biases'].append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units'])], stddev=float(self.conf['biases_std'])), name = 'b%d' % i))
		
			#output layer, initialise as zero
			nnet_dict['weights'].append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), self.conf['num_labels']]), name = 'Wout'))
			nnet_dict['biases'].append(tf.Variable(tf.zeros([self.conf['num_labels']]), name = 'bout'))
		
			#the state prior probabilities
			nnet_dict['state_prior'] = tf.Variable(tf.ones([self.conf['num_labels']]), trainable=False, name = 'priors')
			
			#saver object that saves all the neural net parameters
			nnet_dict['global_saver'] = tf.train.Saver([nnet_dict['state_prior']] + nnet_dict['weights'] + nnet_dict['biases'])
			
			#saver that stores everything but the last hidden layer this is needed to restore the variables when a layer is added in the initialisation
			nnet_dict['prev_saver'] = tf.train.Saver(nnet_dict['weights'][0:num_layers-1] + nnet_dict['biases'][0:num_layers-1] + [nnet_dict['weights'][num_layers], nnet_dict['biases'][num_layers]])
		
		return graph, nnet_dict
	
	#expand the graph for training
	#	graph: basic graph created with create_graph
	#	nnet_dict: neural net created with create_graph
	#	conf: configuration
	def expand_graph_train(self, graph, nnet_dict, conf):
		with graph.as_default():		
			
			#output targets
			nnet_dict['labels'] = tf.placeholder(tf.float32)
		
			#define the derivatives of the weights and biases in a list
			nnet_dict['dweights'] = []
			nnet_dict['dbiases'] = []
			
			#input layer
			nnet_dict['dweights'].append(tf.Variable(tf.zeros([self.conf['input_dim']*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])]), name = 'dWin'))
			nnet_dict['dbiases'].append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units'])]), name = 'dbin'))
			
			#hidden layer (only one updated at a time so only one needed)
			for i in range(len(nnet_dict['weights'])-2):
				nnet_dict['dweights'].append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])]), name = 'dW%d' % i))
				nnet_dict['dbiases'].append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units'])]), name = 'db%d' % i))
			
			#output layer, initialise as zero
			nnet_dict['dweights'].append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), self.conf['num_labels']]), name = 'dWout'))
			nnet_dict['dbiases'].append(tf.Variable(tf.zeros([self.conf['num_labels']]), name = 'dbout'))
			
			#the number of frames presented to compute the gradient
			nnet_dict['num_frames'] = tf.Variable(tf.zeros([], dtype=tf.float32), trainable = False, name = 'num_frames')
			#the total loss of the batch
			nnet_dict['batch_loss'] = tf.Variable(tf.zeros([], dtype=tf.float32), trainable = False, name = 'batch_loss')
			#the total number of steps to be taken
			nnet_dict['num_steps'] = tf.Variable(0, trainable = False, name = 'num_steps')
			#the amount of steps already taken
			nnet_dict['global_step'] = tf.Variable(0, trainable=False, name = 'global_step')
			#a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
			nnet_dict['learning_rate_fact'] = tf.Variable(tf.ones([], dtype=tf.float32), trainable = False, name = 'learning_rate_fact')
			
			#compute the learning rate with exponential decay and scale with the learning rate factor
			nnet_dict['learning_rate'] = tf.mul(tf.train.exponential_decay(float(conf['initial_learning_rate']), nnet_dict['global_step'], nnet_dict['num_steps'], float(conf['learning_rate_decay'])), nnet_dict['learning_rate_fact'])
			
			#define the optimizer
			optimizer = tf.train.AdamOptimizer(nnet_dict['learning_rate'])
				
			#define the training computation (forward prop, back prop, update gradients, update params) 
			#compute the logits (output before softmax)
			out = self.model(nnet_dict['data_in'], nnet_dict['weights'][0:len(nnet_dict['weights'])-1], nnet_dict['biases'][0:len(nnet_dict['biases'])-1], float(conf['dropout']))
			logits = tf.matmul(out, nnet_dict['weights'][len(nnet_dict['weights'])-1]) + nnet_dict['biases'][len(nnet_dict['biases'])-1]
			
			#apply softmax and compute loss
			loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, nnet_dict['labels']))/nnet_dict['num_frames']
			
			#do backprop to compute gradients
			gradients = tf.gradients(loss, nnet_dict['weights'] + nnet_dict['biases'])
			
			#accumulate the gradients and make a list of gradients that need to be applied to update the parameters
			gradients_to_apply = []
			nnet_dict['update_gradients'] = []
			nnet_dict['update_loss'] = nnet_dict['batch_loss'].assign(tf.add(nnet_dict['batch_loss'], loss)).op
			for i in range(len(nnet_dict['dweights'])):
				gradients_to_apply.append((nnet_dict['dweights'][i].value(), nnet_dict['weights'][i]))
				gradients_to_apply.append((nnet_dict['dbiases'][i].value(), nnet_dict['biases'][i]))
				nnet_dict['update_gradients'].append(nnet_dict['dweights'][i].assign(tf.add(nnet_dict['dweights'][i], gradients[i])).op)
				nnet_dict['update_gradients'].append(nnet_dict['dbiases'][i].assign(tf.add(nnet_dict['dbiases'][i], gradients[len(nnet_dict['dweights'])+i])).op)
				
			#apply the gradients to update the parameters
			nnet_dict['optimize'] = optimizer.apply_gradients(gradients_to_apply, global_step=nnet_dict['global_step'])
			
			#prediction computation
			nnet_dict['predictions'] = tf.nn.softmax(logits)
			
			#evaluate the model without useing dropout (needed for validation)
			#compute the logits (output before softmax)
			out = self.model(nnet_dict['data_in'], nnet_dict['weights'][0:len(nnet_dict['weights'])-1], nnet_dict['biases'][0:len(nnet_dict['biases'])-1], 1)
			logits = tf.matmul(out, nnet_dict['weights'][len(nnet_dict['weights'])-1]) + nnet_dict['biases'][len(nnet_dict['biases'])-1]
			
			#apply softmax and compute loss
			loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, nnet_dict['labels']))/nnet_dict['num_frames']
			nnet_dict['update_loss_val'] = nnet_dict['batch_loss'].assign(tf.add(nnet_dict['batch_loss'], loss)).op
			nnet_dict['predictions_val'] = tf.nn.softmax(logits)
				
			#create the visualisations							
			#create loss plot
			loss_summary = tf.scalar_summary('loss', nnet_dict['batch_loss'])
			#create a histogram of weights, biases and their gradients
			weight_summaries = []
			bias_summaries = []
			dweight_summaries = []
			dbias_summaries = []
			
			for i in range(len(nnet_dict['dweights'])):
				weight_summaries.append(tf.histogram_summary('W%d' % i, nnet_dict['weights'][i]))
				dweight_summaries.append(tf.histogram_summary('dW%d' % i, nnet_dict['dweights'][i]))
				bias_summaries.append(tf.histogram_summary('b%d' % i, nnet_dict['biases'][i]))
				dbias_summaries.append(tf.histogram_summary('db%d' % i, nnet_dict['dbiases'][i]))
		
			#merge summaries
			nnet_dict['merged_summary'] = tf.merge_summary(weight_summaries + dweight_summaries + bias_summaries + dbias_summaries + [loss_summary])
			#define writers
			nnet_dict['summary_writer'] = tf.train.SummaryWriter(conf['savedir'] + '/summaries-train')
			
			#saver object that saves the training progress
			nnet_dict['saver'] = tf.train.Saver(max_to_keep=int(conf['check_buffer']))
			
			#if we use the validation set to adapt the learning rate, create a saver to checkpoint the last time the validation set was evaluated
			if conf['valid_adapt'] == 'True':
				nnet_dict['val_saver'] = tf.train.Saver(max_to_keep=1)
				
			#initialisation op
			nnet_dict['init_vars'] = tf.initialize_all_variables()	
	
	
	#compute the prior probability of the states. They are used to compute the pseudo likelihoods. The prior is computed by computing the average predictions from a chosen number of utterances
	#	alignments: the state alignments
	#	conf: the training configuration
	def prior(self, alignments, conf):
	
		#define the decoding operation
		graph, nnet_dict = self.create_graph(int(self.conf['num_hidden_layers']))
		
		prior = np.zeros([self.conf['num_labels']])
		
		#accumulate alignments
		for utt in alignments:
			#put the alignments in one-hot encoding
			onehot = (np.arange(self.conf['num_labels']) == alignments[utt][:,None]).astype(np.float32)
			#count the number of occurences
			prior = prior + np.sum(onehot, 0)
		
		#normalise the prior
		prior = np.divide(prior, np.sum(prior))
		
		#start tensorflow session
		with tf.Session(graph=graph) as session:
			#load the nnet
			nnet_dict['global_saver'].restore(session, conf['savedir'] + '/final')
		
			#set the prior in the neural net
			session.run(nnet_dict['state_prior'].assign(prior))
		
			#save the final neural net with priors
			nnet_dict['global_saver'].save(session, conf['savedir'] + '/final-prio')
	
	#creats the validation set
	#	featdir: directory where the validation features are located
	#	alignments: the state alignments
	#	reader_cmvn: reader for the cmvn statistics
	#	utt2spk: mapping from utterance to speaker
	#	returns the validation data and validation labels
	def create_validation(self, reader, alignments, reader_cmvn, utt2spk):

		#create validation set go through all the utterances in the reader
		#define empty validation data
		val_data = np.empty([0,self.conf['input_dim']*(1+2*int(self.conf['context_width']))], dtype=np.float32)
		#define empty validation labels
		val_labels = np.empty([0,0], dtype=np.float32)
		#read the first utterance
		(utt_id, utt_mat, looped) = reader.read_next_utt()
		while not looped:
			if utt_id in alignments:
				#read cmvn stats
				stats = reader_cmvn.read_utt(utt2spk[utt_id])
				#apply cmvn
				utt_mat = apply_cmvn(utt_mat, stats)
		
				#add the spliced utterance to batch			
				val_data = np.append(val_data, splice(utt_mat,int(self.conf['context_width'])), axis=0)			
		
				#add labels to batch
				val_labels = np.append(val_labels, alignments[utt_id])					
			else:
				print('WARNING no alignment for %s, validation set will be smaller' % utt_id)
			
			#read the next utterance
			(utt_id, utt_mat, looped) = reader.read_next_utt()

		#put labels in one hot encoding	
		if len(val_labels) > 0:
			val_labels = (np.arange(self.conf['num_labels']) == val_labels[:,None]).astype(np.float32)
			
		return val_data, val_labels

	#checks the performance on the validation set goes back in training if performance is worse than previous step
	#	session: the tensorflow session
	#	nnet_dict: the neural net
	#	conf: the training configuration
	#	val_data: the data in the validation set
	#	val_label: the labels in the validation set
	#	step: the current step
	#	prev_step: the step where the validation performance was checked last
	#	old_loss: validation loss when the validation performance was checked last
	#	returns True if validation performance was better, False otherwise
	def validation_step(self, session, nnet_dict, conf, val_data, val_labels, step, prev_step, old_loss):
		#initialise accuracy 	
		p = 0
	
		#tell the neural net how many frames are in the entire batch
		nframes = val_data.shape[0]
		session.run(nnet_dict['num_frames'].assign(nframes))
	
		#feed the batch in a number of minibatches to the neural net and accumulate the loss
		i = 0
		while i < nframes:
			#prepare the data
			if conf['mini_batch_size'] == '-1':
				end_point = nframes
			else:
				end_point = min(i+int(conf['mini_batch_size']), nframes)
			
			feed_dict = {nnet_dict['data_in'] : val_data[i:end_point,:], nnet_dict['labels'] : val_labels[i:end_point,:]}
	
			#accumulate loss and get predictions
			pl, _ = session.run([nnet_dict['predictions_val'], nnet_dict['update_loss_val']], feed_dict = feed_dict)
			#update accuracy
			p += accuracy(pl, val_labels[i:end_point,:])
			#update iterator
			i = end_point
	
		#get the accumulated loss		
		l_val = nnet_dict['batch_loss'].eval(session=session)
		
		#reinitialise the accumulated loss
		tf.initialize_variables([nnet_dict['batch_loss']]).run(session=session)
		
		print("validation loss = %f, validation accuracy = %.1f%%" % (l_val, p/nframes))
		
		#if we're using the validatio set to control the learning rate compare the current validation loss with the one
		if conf['valid_adapt'] == 'True':
			#check if validation loss is lower than previously
			if l_val < old_loss:
				#if performance is better, checkpoint and move on
				nnet_dict['val_saver'].save(session,conf['savedir'] + '/validation/validation-checkpoint')
				#update old loss
				old_loss = l_val
				#set last step that validation has been performed 
				prev_step = step
				return True, prev_step, old_loss
			else:
				#go back to the point where the validation set was previously evaluated
				nnet_dict['val_saver'].restore(session, conf['savedir'] + '/validation/validation-checkpoint')
			
				print('performance on validation set is worse, retrying with halved learning rate')
				#half the learning rate
				session.run(nnet_dict['learning_rate_fact'].assign(nnet_dict['learning_rate_fact'].eval(session = session)/2))
		
				#go back in the dataset to the previous point
				num_utt = 0
				while num_utt < int(conf['batch_size'])*(step-prev_step):
					utt_id = reader.read_previous_scp()
					if utt_id in alignments:
						num_utt = num_utt + 1
				
				#set the step back to the previous point 
				step = prev_step
				session.run(nnet_dict['global_step'].assign(step))
				
				#save the net with adjusted learing_rate_fact	
				nnet_dict['val_saver'].save(session,conf['savedir'] + '/validation/validation-checkpoint')
				
				return False, prev_step, old_loss
		else:
			return True, prev_step, old_loss
			
	#does the training step: forward, backward pass and update parameters
	#	session: the tensorflow session
	#	nnet_dict: the neural net
	#	conf: the training configuration
	#	step: the current step
	#	batch_data: the data for this batch
	#	batch_label: the labels for this batch
	def training_step(self, session, nnet_dict, conf, step, batch_data, batch_labels):
		#tell the neural net how many frames are in the entire batch
		nframes = batch_data.shape[0]
		session.run(nnet_dict['num_frames'].assign(nframes))
	
		#feed the batch in a number of minibatches to the neural net and accumulate the gradients and loss (we do it this way to limit memory usage)	
		finished = False
		p = 0
		while not finished:
		
			# prepare data
			if batch_data.shape[0] > int(conf['mini_batch_size']) and conf['mini_batch_size'] != '-1':
				feed_labels = batch_labels[0:int(conf['mini_batch_size']),:]
				feed_dict = {nnet_dict['data_in'] : batch_data[0:int(conf['mini_batch_size']),:], nnet_dict['labels'] : feed_labels}
				batch_data = batch_data[int(conf['mini_batch_size']):batch_data.shape[0],:]
				batch_labels = batch_labels[int(conf['mini_batch_size']):batch_labels.shape[0],:]
			else:
				feed_labels = batch_labels
				feed_dict = {nnet_dict['data_in'] : batch_data, nnet_dict['labels'] : feed_labels}
				finished = True
				
			#do forward-backward pass and update gradients	
			out = session.run([nnet_dict['predictions'], nnet_dict['update_loss']] + nnet_dict['update_gradients'], feed_dict=feed_dict)
			#update batch accuracy
			p += accuracy(out[0], feed_labels)
	
		#write summaries to disk so Tensorboard can read them
		if conf['visualise'] == 'True':
			nnet_dict['summary_writer'].add_summary(nnet_dict['merged_summary'].eval(session=session), global_step = step)
			
		#do the optimization operation
		session.run(nnet_dict['optimize'])
			
		print("step %d: training loss = %f, accuracy = %.1f%%, learning rate = %f, #frames in batch = %d" % (step + 1, nnet_dict['batch_loss'].eval(session=session), p/nframes, nnet_dict['learning_rate'].eval(session=session), nframes))
	
		#reinitlialize the gradients, loss and prediction accuracy
		tf.initialize_variables(nnet_dict['dweights'] + nnet_dict['dbiases'] + [nnet_dict['batch_loss']]).run(session=session)
	
	# Train the neural network with stochastic gradient descent 
	#	featdir: directory where the features are located
	#	alignments: dictionary containing the state alignments
	#	utt2spk: mapping from utterance to speaker
	def train(self, featdir, alignments, utt2spk, conf):
		
		#clear summaries	
		if os.path.isdir(conf['savedir'] + '/summaries-train'):
			shutil.rmtree(conf['savedir'] + '/summaries-train')		
		
		#if starting step is final skip training
		if conf['starting_step'] != 'final':	
			
			#open cmvn statistics reader
			reader_cmvn = kaldi_io.KaldiReadIn(featdir + '/cmvn.scp')
			
			#open feature reader for training data
			reader = kaldi_io.KaldiReadIn(featdir + '/feats_shuffled.scp')
			
			#create the validation set
			if int(conf['valid_size']) > 0:
				val_data, val_labels = self.create_validation(reader.split(int(conf['valid_size'])), alignments, reader_cmvn, utt2spk)
						
			#go to the point in the database where the training was at checkpoint
			num_utt = 0
			while num_utt < int(conf['batch_size'])*int(conf['starting_step']):
				utt_id = reader.read_next_scp()
				if utt_id in alignments:
					num_utt = num_utt + 1
			
			#check to see if we need to initialise the neural net or load a checkpointed one
			if conf['starting_step'] == '0':		
				num_layers = 1	
				graph, nnet_dict = self.create_graph(num_layers)
				self.expand_graph_train(graph, nnet_dict, conf)
				session = tf.Session(graph=graph)
				session.run(nnet_dict['init_vars'])
				step = 0
			else:
				num_layers = min(int(int(conf['starting_step'])/int(conf['add_layer_period']))+1, int(self.conf['num_hidden_layers']))
				graph, nnet_dict = self.create_graph(num_layers)
				self.expand_graph_train(graph, nnet_dict, conf)
				session = tf.Session(graph=graph)
				nnet_dict['saver'].restore(session, conf['savedir'] + '/training/model-' + conf['starting_step'])
				step = int(conf['starting_step'])				

			#calculate number of steps
			nsteps =  int(int(conf['num_epochs']) * len(alignments) / int(conf['batch_size']))
			print('starting training, total number of steps = %d' % nsteps)
		
			#set the number of steps
			session.run(nnet_dict['num_steps'].assign(nsteps))
		
			#compute the first validation perfomance
			if int(conf['valid_size']) > 0:
				retry_count = 0
				_, prev_step, old_loss = self.validation_step(session, nnet_dict, conf, val_data, val_labels, step, 0, float('inf'))
		
			#flag to see if all layers have been added
			if num_layers == int(self.conf['num_hidden_layers']):
				nnet_complete = True
			else:
				nnet_complete = False
			
			#loop over number of steps
			while step < nsteps:
			
				#add a layer
				if step % int(conf['add_layer_period']) == 0 and step != 0 and not nnet_complete:
					num_layers += 1
					print('adding a layer to the nearal net, current number of layers = %d' % (num_layers))
					#save the current neural net
					nnet_dict['global_saver'].save(session, conf['savedir'] + '/training/model-addlayer')
					#close the session
					session.close()
					#create a new graph with an extra layer
					graph, nnet_dict = self.create_graph(num_layers)
					self.expand_graph_train(graph, nnet_dict, conf)
					#open a session
					session = tf.Session(graph=graph)
					#initialise the graph
					session.run(nnet_dict['init_vars'])
					#load the previous neural net
					nnet_dict['prev_saver'].restore(session, conf['savedir'] + '/training/model-addlayer')
					#reinitialise the softmax
					tf.initialize_variables([nnet_dict['weights'][num_layers], nnet_dict['biases'][num_layers]]).run(session = session)
					#compute the first validation perfomance
					if int(conf['valid_size']) > 0:					
						retry_count = 0
						_, prev_step, old_loss = self.validation_step(session, nnet_dict, conf, val_data, val_labels, step, prev_step, old_loss = float('inf'))
					if num_layers == int(self.conf['num_hidden_layers']):
						nnet_complete = True
						#visualize the complete graph
						if conf['visualise']=='True':
							nnet_dict['summary_writer'].add_graph(session.graph_def)
				
				#training step
				#create a training batch 
				(batch_data, batch_labels) = create_batch(reader, reader_cmvn, alignments, utt2spk, self.conf['input_dim'], int(self.conf['context_width']), self.conf['num_labels'], int(conf['batch_size']))
				#do the training step
				self.training_step(session, nnet_dict, conf, step, batch_data, batch_labels)
				
				#check performance on validation set if we're at a validation frequency step with a complete net or an incomplete net where a layer is going to be added in the next step
				if int(conf['valid_size']) > 0 and ((step % int(conf['valid_frequency']) == 0 and nnet_complete) or ((step + 1) % int(conf['add_layer_period']) == 0 and not nnet_complete)):
					sucess, prev_step, old_loss = self.validation_step(session, nnet_dict, conf, val_data, val_labels, step, prev_step, old_loss)
					if sucess:
						#reset retry counter
						retry_count = 0
					else:
						#increment retry counter
						retry_count += 1
						#if the maximum number of retries has been reached, terminate training
						if retry_count == int(conf['valid_retries']):
							print('WARNING: terminating learning (early stopping)')
							break
						else:
							continue
					
				
				#save the neural net if at checkpoint
				if step % int(conf['check_freq']) == 0:
					nnet_dict['saver'].save(session, conf['savedir'] + '/training/model', global_step=step)
				
				#increment the step
				step += 1
				
			#save the final neural net
			nnet_dict['global_saver'].save(session, conf['savedir'] + '/final')
			
			#close the tf session 
			session.close()

		#compute the state prior probabilities
		self.prior(alignments, conf)

	#compute pseudo likelihoods for a set of utterances
	#	featdir: directory where the features are located
	#	utt2spk: mapping from utterance to speaker 
	#	savedir: location of the neural net
	#	decodir: location wher output should be stored
	def decode(self, featdir, utt2spk, savedir, decodedir):
		
		#define the decoding operation
		graph, nnet_dict = self.create_graph(int(self.conf['num_hidden_layers']))
		with graph.as_default():
			out = self.model(nnet_dict['data_in'], nnet_dict['weights'][0:len(nnet_dict['weights'])-1], nnet_dict['biases'][0:len(nnet_dict['biases'])-1], 1)
			logits = tf.matmul(out, nnet_dict['weights'][len(nnet_dict['weights'])-1]) + nnet_dict['biases'][len(nnet_dict['biases'])-1]
			predictions = tf.nn.softmax(logits)
			
		#open feature reader
		reader = kaldi_io.KaldiReadIn(featdir + '/feats.scp')
		#remove ark file if it allready exists
		if os.path.isfile(decodedir + '/feats.ark'):
			os.remove(decodedir + '/feats.ark')
		#open cmvn statistics reader
		reader_cmvn = kaldi_io.KaldiReadIn(featdir + '/cmvn.scp')			
		#open likelihood writer
		writer = kaldi_io.KaldiWriteOut(decodedir + '/feats.scp')
		
		#start tensorflow session
		with tf.Session(graph=graph) as session:
			#load the final neural net with priors
			nnet_dict['global_saver'].restore(session, savedir + '/final-prio')
		
			#get the state prior out of the graph
			prior = nnet_dict['state_prior'].eval()
		
			#feed the utterances one by one to the neural net
			while True:
				(utt_id, utt_mat, looped) = reader.read_next_utt()
				if looped:
					break
				
				#read the cmvn stats
				stats = reader_cmvn.read_utt(utt2spk[utt_id])
				#apply cmvn
				utt_mat = apply_cmvn(utt_mat, stats)
				
				#prepare data
				feed_dict = {nnet_dict['data_in'] : splice(utt_mat,int(self.conf['context_width']))}
				#compute predictions
				p = session.run(predictions, feed_dict=feed_dict)
				#apply prior to predictions
				p = np.divide(p,prior)
				#write the pseudo-likelihoods in kaldi feature format
				writer.write_next_utt(decodedir + '/feats.ark', utt_id, p)
			
		#close the writer
		writer.close()


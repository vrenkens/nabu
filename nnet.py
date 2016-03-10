import kaldi_io

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import gzip
import shutil
import os
import copy

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
#	log: log file
def create_batch(reader, reader_cmvn, alignments, utt2spk, input_dim, context_width, num_labels, batch_size, log):
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
			log.write('WARNING no alignment for %s\n' % utt_id)
			
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

		#apply dropout	
		if dropout<1:
			data = tf.nn.dropout(data, dropout)
		#apply l2 normalisation
		if self.conf['l2_norm'] == 'True':
			data = tf.nn.l2_normalize(data,1)*np.sqrt(float(self.conf['num_hidden_units']))				
	
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

	def create_graph(self):
		nnet = {}
		graph = tf.Graph()
		with graph.as_default():
			#input data
			nnet['data_in'] = tf.placeholder(tf.float32, shape = [None, self.conf['input_dim']*(1+2*int(self.conf['context_width']))])
		
			#define weights, biases and their derivatives lists
			nnet['weights'] = []
			nnet['biases'] = []
		
			#input layer, initialise as random normal
			nnet['weights'].append(tf.Variable(tf.random_normal([self.conf['input_dim']*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])], stddev=float(self.conf['weights_std'])), name = 'W0'))
			nnet['biases'].append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units'])], stddev=float(self.conf['biases_std'])), name = 'b0'))
		
			#hidden layers, initialise as random normal
			for i in range(int(self.conf['num_hidden_layers'])-1):
				nnet['weights'].append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])], stddev=float(self.conf['weights_std'])), name = 'W%d' % (i+1)))
				nnet['biases'].append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units'])], stddev=float(self.conf['biases_std'])), name = 'b%d' % (i+1)))
		
			#output layer, initialise as zero
			nnet['weights'].append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), self.conf['num_labels']]), name = 'W%d' % int(self.conf['num_hidden_layers'])))
			nnet['biases'].append(tf.Variable(tf.zeros([self.conf['num_labels']]), name = 'b%d' % int(self.conf['num_hidden_layers'])))
		
			#the state prior probabilities
			nnet['state_prior'] = tf.Variable(tf.ones([self.conf['num_labels']]), trainable=False, name = 'priors')
			
			#saver object that saves all the neural net parameters
			nnet['global_saver'] = tf.train.Saver([nnet['state_prior']] + nnet['weights'] + nnet['biases'])
		
		return graph, nnet
	
	#Initialise the neural net. We start with a neural net with one hidden layer, we train the hidden layer and the softmax for a couple of steps. We then add a new hidden layer and reinitialise the softmax layer. We then train the added layer and the softmax. We do this until the correct number of hidden layers is reached 
	#	featdir: directory where the features are located
	#	alignments: dictionary containing the state alignments
	#	utt2spk: mapping from utterance to speaker
	def initialise(self, featdir, alignments, utt2spk, conf):
	
		#clear summaries	
		if os.path.isdir(conf['savedir'] + '/summaries-init'):
			shutil.rmtree(conf['savedir'] + '/summaries-init')
		
		#define the initialisation computation for all the number of layers (needed for layer by layer initialisation). In the initialisation 
		graph, nnet = self.create_graph()
		with graph.as_default():
		
			#output targets
			labels = tf.placeholder(tf.float32)
		
			#define the derivatives of the weights and biases in a list
			dweights = []
			dbiases = []
			
			#input layer
			dweights.append(tf.Variable(tf.zeros([self.conf['input_dim']*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])]), name = 'dW0'))
			dbiases.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units'])]), name = 'db0'))
			
			#hidden layer (only one updated at a time so only one needed)
			for i in range(int(self.conf['num_hidden_layers'])-1):
				dweights.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])]), name = 'dW%d' % (i+1)))
				dbiases.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units'])]), name = 'db%d' % (i+1)))
			
			#output layer, initialise as zero
			dweights.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), self.conf['num_labels']]), name = 'dW%d' % int(self.conf['num_hidden_layers'])))
			dbiases.append(tf.Variable(tf.zeros([self.conf['num_labels']]), name = 'db%d' % int(self.conf['num_hidden_layers'])))
			
			#the number of frames presented to compute the gradient
			num_frames = tf.Variable(tf.zeros([], dtype=tf.float32), trainable = False, name = 'num_frames')
			#the total loss of the batch
			batch_loss = tf.Variable(tf.zeros([], dtype=tf.float32), trainable = False, name = 'batch_loss')
			
			#define the optimizer, with or without momentum, no exponential decay for initialisation
			if float(conf['momentum']) > 0:
				optimizer = tf.train.MomentumOptimizer(float(conf['learning_rate_init']),float(conf['momentum']))
			else:
				optimizer = tf.train.GradientDescentOptimizer(float(conf['learning_rate_init']))
		
			loss = []
			optimize = []
			update_gradients = []
			for num_layers in range(int(self.conf['num_hidden_layers'])):
		
				#compute the logits (output before softmax)
				out = self.model(nnet['data_in'], nnet['weights'][0:num_layers+1], nnet['biases'][0:num_layers+1], conf['dropout'])
				logits = tf.matmul(out, nnet['weights'][len(nnet['weights'])-1]) + nnet['biases'][len(nnet['biases'])-1]
			
				#apply softmax and compute loss
				loss.append(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))/num_frames)
				
				#compute the gradients
				gradients = tf.gradients(loss[num_layers], [nnet['weights'][num_layers], nnet['biases'][num_layers], nnet['weights'][len(nnet['weights'])-1], nnet['biases'][len(nnet['biases'])-1]])
				
				#operations to accumulate the gradients
				update_gradients.append([dweights[num_layers].assign(tf.add(dweights[num_layers], gradients[0])).op])
				update_gradients[num_layers].append(dbiases[num_layers].assign(tf.add(dbiases[num_layers], gradients[1])).op)
				update_gradients[num_layers].append(dweights[len(dweights)-1].assign(tf.add(dweights[len(dweights)-1], gradients[2])).op)
				update_gradients[num_layers].append(dbiases[len(dbiases)-1].assign(tf.add(dbiases[len(dbiases)-1], gradients[3])).op)
				
				#operation to accumulate the loss
				update_gradients[num_layers].append(batch_loss.assign(tf.add(batch_loss, loss[num_layers])).op)
			
				#list of gradients that will be used to update the parameters
				gradients_to_apply = [(dweights[num_layers].value(), nnet['weights'][num_layers]), (dweights[len(dweights)-1].value(), nnet['weights'][len(dweights)-1]), (dbiases[num_layers].value(), nnet['biases'][num_layers]), (dbiases[len(dbiases)-1].value(), nnet['biases'][len(dbiases)-1])]
			
				#operation to apply the computed gradients
				optimize.append(optimizer.apply_gradients(gradients_to_apply))
				
			#create the visualisations							
			#create loss plot
			loss_summary = tf.scalar_summary('loss', batch_loss)
			#create a histogram of weights and biases
			weight_summaries = []
			bias_summaries = []
			dweight_summaries = []
			dbias_summaries = []
			
			for i in range(int(self.conf['num_hidden_layers'])+1):
				weight_summaries.append(tf.histogram_summary('W%d' % i, nnet['weights'][i]))
				dweight_summaries.append(tf.histogram_summary('dW%d' % i, dweights[i]))
				bias_summaries.append(tf.histogram_summary('b%d' % i, nnet['biases'][i]))
				dbias_summaries.append(tf.histogram_summary('db%d' % i, dbiases[i]))
		
			#merge summaries
			merged_summary = tf.merge_summary(weight_summaries + dweight_summaries + bias_summaries + dbias_summaries + [loss_summary])
			#define writers
			summary_writer = tf.train.SummaryWriter(conf['savedir'] + '/summaries-init')
	
	
		#open feature reader
		reader = kaldi_io.KaldiReadIn(featdir + '/feats_shuffled.scp')
		#open the cmvn statistics reader
		reader_cmvn = kaldi_io.KaldiReadIn(featdir + '/cmvn.scp')	
		#open log
		log = open(conf['savedir'] + '/init.log', 'w')
		
		#start tensorflow session
		with tf.Session(graph=graph) as session:
						
			#initialize the variables
			tf.initialize_all_variables().run()
			
			if conf['visualise']=='True':
				summary_writer.add_graph(session.graph_def)
			
			#do layer by layer initialization
			for num_layers in range(int(self.conf['num_hidden_layers'])):
			
				#reinitialize the softmax
				tf.initialize_variables([nnet['weights'][int(self.conf['num_hidden_layers'])], nnet['biases'][int(self.conf['num_hidden_layers'])]]).run()
				
				for step in range(int(conf['init_steps'])):
					
					#create a batch 
					(batch_data, batch_labels) = create_batch(reader, reader_cmvn, alignments, utt2spk, self.conf['input_dim'], int(self.conf['context_width']), self.conf['num_labels'], int(conf['batch_size']), log)
					
					#tell the neural net how many frames are in the entire batch
					nframes = batch_data.shape[0]
					session.run(num_frames.assign(nframes))
					
					#feed the batch in a number of minibatches to the neural net and accumulate the gradients and loss
					finished = False
					while not finished:
			
						#prepare nnet data
						if batch_data.shape[0] > int(conf['mini_batch_size']) and conf['mini_batch_size'] != '-1':
							feed_dict = {nnet['data_in'] : batch_data[0:int(conf['mini_batch_size']),:], labels : batch_labels[0:int(conf['mini_batch_size']),:]}
							batch_data = batch_data[int(conf['mini_batch_size']):batch_data.shape[0],:]
							batch_labels = batch_labels[int(conf['mini_batch_size']):batch_labels.shape[0],:]
						else:
							feed_dict = {nnet['data_in'] : batch_data, labels : batch_labels}
							finished = True
								
						#do forward backward pass and update gradients					
						session.run(update_gradients[num_layers], feed_dict=feed_dict)
						
					#write the summaries to disk so Tensorboard can read them 
					if conf['visualise'] == 'True':
						summary_writer.add_summary(merged_summary.eval(), global_step = step + int(conf['init_steps'])*num_layers)
					
					#do the appropriate optimization operation
					session.run(optimize[num_layers])	
					print("initialization step %d/%d, #layers %d: training loss = %f" % (step + 1, int(conf['init_steps']), num_layers+1, batch_loss.eval()))
					
					#reinitlialize the gradients, loss and prediction accuracy
					tf.initialize_variables(dweights + dbiases + [batch_loss]).run()
					
				#save the initialised neural net
				nnet['global_saver'].save(session, conf['savedir'] + '/init')
			
			#close the log
			log.close()
			
			#close the summary writer so all the summaries still in the pipe are written to disk
			summary_writer.close()
	
	#compute the prior probability of the states. They are used to compute the pseudo likelihoods. The prior is computed by computing the average predictions from a chosen number of utterances
	#	featdir: directory where the features are located
	#	utt2spk: mapping from utterance to speaker
	def prior(self, featdir, utt2spk, conf):
	
		#define the decoding operation
		graph, nnet = self.create_graph()
		with graph.as_default():
			out = self.model(nnet['data_in'], nnet['weights'][0:len(nnet['weights'])-1], nnet['biases'][0:len(nnet['biases'])-1], 1)
			logits = tf.matmul(out, nnet['weights'][len(nnet['weights'])-1]) + nnet['biases'][len(nnet['biases'])-1]
			predictions = tf.nn.softmax(logits)
	
		#open feature reader
		reader = kaldi_io.KaldiReadIn(featdir + '/feats_shuffled.scp')
		#open cmvn statistics reader
		reader_cmvn = kaldi_io.KaldiReadIn(featdir + '/cmvn.scp')
		
		#start tensorflow session
		with tf.Session(graph=graph) as session:
			#load the final neural net
			nnet['global_saver'].restore(session, conf['savedir'] + '/final')
		
			#create the batch to compute the prior
			batch_data = np.empty([0,self.conf['input_dim']*(1+2*int(self.conf['context_width']))], dtype=np.float32)
			num_utt = 0
			for _ in range(int(conf['ex_prio'])):
				#read utterance
				(utt_id, utt_mat, looped) = reader.read_next_utt()
				#read cmvn stats
				stats = reader_cmvn.read_utt(utt2spk[utt_id])
				#apply cmvn stats
				utt_mat = apply_cmvn(utt_mat, stats)
			
				if looped:
					print('WARNING: not enough utterances to compute the prior')
					break
			
				#add the spiced utterance to batch			
				batch_data = np.append(batch_data, splice(utt_mat,int(self.conf['context_width'])), axis=0)
		
			#initialise the prior as zeros
			prior = np.zeros(self.conf['num_labels'])
			finished = False
			while not finished:
				# prepare data
				if batch_data.shape[0] > int(conf['mini_batch_size']) and conf['mini_batch_size'] != '-1':
					feed_dict = {nnet['data_in'] : batch_data[0:int(conf['mini_batch_size']),:]}
					batch_data = batch_data[int(conf['mini_batch_size']):batch_data.shape[0],:]
				else:
					feed_dict = {nnet['data_in'] : batch_data}
					finished = True
	
				#compute the predictions
				p = session.run(predictions, feed_dict=feed_dict)

				#accumulate the predictions in the prior
				prior += np.sum(p,0)
		
			#normalise the prior
			prior = np.divide(prior, np.sum(prior))
		
			#set the prior in the neural net
			session.run(nnet['state_prior'].assign(prior))
		
			#save the final neural net with priors
			nnet['global_saver'].save(session, conf['savedir'] + '/final-prio')
	
	
	# Train the neural network with stochastic gradient descent 
	#	featdir: directory where the features are located
	#	alignments: dictionary containing the state alignments
	#	utt2spk: mapping from utterance to speaker
	def train(self, featdir, alignments, utt2spk, conf):
		#clear summaries	
		if os.path.isdir(conf['savedir'] + '/summaries-train'):
			shutil.rmtree(conf['savedir'] + '/summaries-train')
			
		graph, nnet = self.create_graph()
		with graph.as_default():		
			
			#output targets
			labels = tf.placeholder(tf.float32)
		
			#define the derivatives of the weights and biases in a list
			dweights = []
			dbiases = []
			
			#input layer
			dweights.append(tf.Variable(tf.zeros([self.conf['input_dim']*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])]), name = 'dW0'))
			dbiases.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units'])]), name = 'db0'))
			
			#hidden layer (only one updated at a time so only one needed)
			for i in range(int(self.conf['num_hidden_layers'])-1):
				dweights.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])]), name = 'dW%d' % (i+1)))
				dbiases.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units'])]), name = 'db%d' % (i+1)))
			
			#output layer, initialise as zero
			dweights.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), self.conf['num_labels']]), name = 'dW%d' % int(self.conf['num_hidden_layers'])))
			dbiases.append(tf.Variable(tf.zeros([self.conf['num_labels']]), name = 'db%d' % int(self.conf['num_hidden_layers'])))
			
			#the number of frames presented to compute the gradient
			num_frames = tf.Variable(tf.zeros([], dtype=tf.float32), trainable = False, name = 'num_frames')
			#the total loss of the batch
			batch_loss = tf.Variable(tf.zeros([], dtype=tf.float32), trainable = False, name = 'batch_loss')
			#the total number of steps to be taken
			num_steps = tf.Variable(0, trainable = False, name = 'num_steps')
			#the amount of steps already taken
			global_step = tf.Variable(0, trainable=False, name = 'global_step')
			#a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
			learning_rate_fact = tf.Variable(tf.ones([], dtype=tf.float32), trainable = False, name = 'learning_rate_fact')
			
			#compute the learning rate with exponential decay and scale with the learning rate factor
			learning_rate = tf.mul(tf.train.exponential_decay(float(conf['initial_learning_rate']), global_step, num_steps, float(conf['learning_rate_decay'])), learning_rate_fact)
			
			#define the optimizer, with or without momentum
			if float(conf['momentum']) > 0:
				optimizer = tf.train.MomentumOptimizer(learning_rate,float(conf['momentum']))
			else:
				optimizer = tf.train.GradientDescentOptimizer(learning_rate)
				
			#define the training computation (forward prop, back prop, update gradients, update params) 
			#compute the logits (output before softmax)
			out = self.model(nnet['data_in'], nnet['weights'][0:len(nnet['weights'])-1], nnet['biases'][0:len(nnet['biases'])-1], conf['dropout'])
			logits = tf.matmul(out, nnet['weights'][len(nnet['weights'])-1]) + nnet['biases'][len(nnet['biases'])-1]
			
			#apply softmax and compute loss
			loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))/num_frames
			
			#do backprop to compute gradients
			gradients = tf.gradients(loss,nnet['weights'] + nnet['biases'])	
			
			#accumulate the gradients and make a list of gradients that need to be applied to update the parameters
			gradients_to_apply = []
			update_gradients = []
			update_loss = batch_loss.assign(tf.add(batch_loss, loss)).op
			for i in range(len(dweights)):
				gradients_to_apply.append((dweights[i].value(), nnet['weights'][i]))
				gradients_to_apply.append((dbiases[i].value(), nnet['biases'][i]))
				update_gradients.append(dweights[i].assign(tf.add(dweights[i], gradients[i])).op)
				update_gradients.append(dbiases[i].assign(tf.add(dbiases[i], gradients[len(dweights)+i])).op)
				
			#apply the gradients to update the parameters
			optimize = optimizer.apply_gradients(gradients_to_apply, global_step=global_step)
			
			#prediction computation
			predictions = tf.nn.softmax(logits)
				
			#create the visualisations							
			#create loss plot
			loss_summary = tf.scalar_summary('loss', batch_loss)
			#create a histogram of weights and biases
			weight_summaries = []
			bias_summaries = []
			dweight_summaries = []
			dbias_summaries = []
			
			for i in range(int(self.conf['num_hidden_layers'])+1):
				weight_summaries.append(tf.histogram_summary('W%d' % i, nnet['weights'][i]))
				dweight_summaries.append(tf.histogram_summary('dW%d' % i, dweights[i]))
				bias_summaries.append(tf.histogram_summary('b%d' % i, nnet['biases'][i]))
				dbias_summaries.append(tf.histogram_summary('db%d' % i, dbiases[i]))
		
			#merge summaries
			merged_summary = tf.merge_summary(weight_summaries + dweight_summaries + bias_summaries + dbias_summaries + [loss_summary])
			#define writers
			summary_writer = tf.train.SummaryWriter(conf['savedir'] + '/summaries-train')
			
			#saver object that saves the training progress
			saver = tf.train.Saver(max_to_keep=int(conf['check_buffer']))
			
			#if we use the validation set to adapt the learning rate, create a saver to checkpoint the last time the validation set was evaluated
			if conf['valid_adapt'] == 'True':
				val_saver = tf.train.Saver(max_to_keep=1)
			
			
		if conf['starting_step'] != 'final':	
			#open log
			log = open(conf['savedir'] + '/train.log', 'w')		
			#open feature reader for validation data
			reader = kaldi_io.KaldiReadIn(featdir + '/feats_validation.scp')
			#open cmvn statistics reader
			reader_cmvn = kaldi_io.KaldiReadIn(featdir + '/cmvn.scp')

			#create validation set go through all the utterances in feats_validation.scp
			val_data = np.empty([0,self.conf['input_dim']*(1+2*int(self.conf['context_width']))], dtype=np.float32)
			val_labels = np.empty([0,0], dtype=np.float32)
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
					log.write('WARNING no alignment for %s, validation set will be smaller\n' % utt_id)
			
				(utt_id, utt_mat, looped) = reader.read_next_utt()

			#put labels in one hot encoding	
			val_labels = (np.arange(self.conf['num_labels']) == val_labels[:,None]).astype(np.float32)	
		
			#open feature reader for training data
			reader = kaldi_io.KaldiReadIn(featdir + '/feats_shuffled.scp')
				
			#go to the initial line (start at point after initilaization and #steps allready taken)
			num_utt = 0
			while num_utt < int(conf['batch_size'])*(int(conf['starting_step'])+int(conf['init_steps'])*int(self.conf['num_hidden_layers'])):
				utt_id = reader.read_next_scp()
				if utt_id in alignments:
					num_utt = num_utt + 1

			#initialise the neural net
			if conf['starting_step'] == '-1':
				self.initialise(featdir, alignments, utt2spk, conf)
				step = 0
			
			#start tensorflow session
			with tf.Session(graph=graph) as session:
				if conf['starting_step'] == '-1':			
					tf.initialize_all_variables().run()
					nnet['global_saver'].restore(session, conf['savedir'] + '/init')
				
					#save the initial neural net
					saver.save(session, conf['savedir'] + '/training/model', global_step = 0)
				else:
					saver.restore(session, conf['savedir'] + '/training/model-' + conf['starting_step'])
					step = int(conf['starting_step'])
		
				#visualize the graph
				if conf['visualise']=='True':
					summary_writer.add_graph(session.graph_def)
					summary_writer
			
				#calculate number of steps
				nsteps =  int(int(conf['num_epochs']) * len(alignments) / int(conf['batch_size']))
			
				#set the number of steps
				session.run(num_steps.assign(nsteps))
			
				if conf['valid_adapt'] == 'True':
					#initialise old loss to infinity
					old_loss = float('inf')
					#initialise the number of retries (number of consecutive times the training had to go back with half learning rate)
					retry_count = 0
			
				#loop over number of steps
				while step < nsteps:
			
					#check performance on evaluation set
					if val_data.shape[0] > 0 and step % int(conf['valid_frequency']) == 0:
						#renitialise accuracy 	
						p = 0
					
						#tell the neural net how many frames are in the entire batch
						nframes = val_data.shape[0]
						session.run(num_frames.assign(nframes))
					
						#feed the batch in a number of minibatches to the neural net and accumulate the loss
						i = 0
						while i < nframes:
							#prepare the data
							if conf['mini_batch_size'] == '-1':
								end_point = nframes
							else:
								end_point = min(i+int(conf['mini_batch_size']), nframes)
							
							feed_dict = {nnet['data_in'] : val_data[i:end_point,:], labels : val_labels[i:end_point,:]}
					
							#accumulate loss and get predictions
							pl, _ = session.run([predictions, update_loss], feed_dict = feed_dict)
							#update accuracy
							p += accuracy(pl, val_labels[i:end_point,:])
							#update iterator
							i = end_point
					
						#get the accumulated loss		
						l_val = batch_loss.eval()
						#reinitialise the accumulated loss
						tf.initialize_variables([batch_loss]).run()
						print("validation loss = %f, validation accuracy = %.1f%%" % (l_val, p/nframes))
						#check if validation loss is lower than previously
						if l_val < old_loss:
							#if performance is better, checkpoint and move on
							val_saver.save(session,conf['savedir'] + '/validation/validation-checkpoint')
							#update old loss
							old_loss = l_val
							#set retry count to 0
							retry_count = 0
						else:
							#go back to the point where the validation set was previously evaluated
							val_saver.restore(session, conf['savedir'] + '/validation/validation-checkpoint')
						
							#if the maximum number of retries has been reached, terminate training
							if retry_count == int(conf['valid_retries']):
								print('WARNING: terminating learning (early stopping)')
								break
							
							print('performance on validation set is worse, retrying with halved learning rate')
							#half the learning rate
							session.run(learning_rate_fact.assign(learning_rate_fact.eval()/2))
							#set the step back to the previous point 
							step = step - int(conf['valid_frequency'])
						
							#go back in the dataset to the previous point
							num_utt = 0
							while num_utt < int(conf['batch_size'])*int(conf['valid_frequency']):
								utt_id = reader.read_previous_scp()
								if utt_id in alignments:
									num_utt = num_utt + 1
						
							#increment the retry count
							retry_count += 1
				
					#create a training batch 
					(batch_data, batch_labels) = create_batch(reader, reader_cmvn, alignments, utt2spk, self.conf['input_dim'], int(self.conf['context_width']), self.conf['num_labels'], int(conf['batch_size']), log)
				
					#tell the neural net how many frames are in the entire batch
					nframes = batch_data.shape[0]
					session.run(num_frames.assign(nframes))
				
					#feed the batch in a number of minibatches to the neural net and accumulate the gradients and loss (we do it this way to limit memory usage)	
					finished = False
					p = 0
					while not finished:
					
						# prepare data
						if batch_data.shape[0] > int(conf['mini_batch_size']) and conf['mini_batch_size'] != '-1':
							feed_labels = batch_labels[0:int(conf['mini_batch_size']),:]
							feed_dict = {nnet['data_in'] : batch_data[0:int(conf['mini_batch_size']),:], labels : feed_labels}
							batch_data = batch_data[int(conf['mini_batch_size']):batch_data.shape[0],:]
							batch_labels = batch_labels[int(conf['mini_batch_size']):batch_labels.shape[0],:]
						else:
							feed_labels = batch_labels
							feed_dict = {nnet['data_in'] : batch_data, labels : feed_labels}
							finished = True
							
						#do forward-backward pass and update gradients	
						out = session.run([predictions, update_loss] + update_gradients, feed_dict=feed_dict)
						#update batch accuracy
						p += accuracy(out[0], feed_labels)
				
					#write summaries to disk so Tensorboard can read them
					if conf['visualise'] == 'True':
						summary_writer.add_summary(merged_summary.eval(), global_step = step)
				
					#do optimization operation (apply the accumulated gradients)	
					session.run(optimize)
					print("step %d/%d: training loss = %f, accuracy = %.1f%%, learning rate = %f, #frames in batch = %d" % (step + 1, nsteps, batch_loss.eval(), p/nframes, learning_rate.eval(), nframes))
				
					#reinitlialize the gradients, loss and prediction accuracy
					tf.initialize_variables(dweights + dbiases + [batch_loss]).run()
				
					#save the neural net if at checkpoint
					if step % int(conf['check_freq']) == 0:
						saver.save(session, conf['savedir'] + '/training/model', global_step=step)
					
					#increment the step
					step += 1
						
				#save the final neural net
				nnet['global_saver'].save(session, conf['savedir'] + '/final')

		#compute the state prior probabilities
		self.prior(featdir, utt2spk, conf)
		
		#close the log 
		log.close()

	#compute pseudo likelihoods for a set of utterances
	#	featdir: directory where the features are located
	#	utt2spk: mapping from utterance to speaker 
	def decode(self, featdir, utt2spk, savedir, decodedir):
		
		#define the decoding operation
		graph, nnet = self.create_graph()
		with graph.as_default():
			out = self.model(nnet['data_in'], nnet['weights'][0:len(nnet['weights'])-1], nnet['biases'][0:len(nnet['biases'])-1], 1)
			logits = tf.matmul(out, nnet['weights'][len(nnet['weights'])-1]) + nnet['biases'][len(nnet['biases'])-1]
			predictions = tf.nn.softmax(logits)
			
		#open feature reader
		reader = kaldi_io.KaldiReadIn(featdir + '/feats.scp')
		#remove ark file if it allready exists
		if os.path.isfile(decodedir + '/feats.ark'):
			os.remove(decodedir + '/feats.ark')
		#open cmvn statistics reader
		reader_cmvn = kaldi_io.KaldiReadIn(featdir + '/cmvn.scp')			
		#open prior writer
		writer = kaldi_io.KaldiWriteOut(decodedir + '/feats.scp')
		
		#start tensorflow session
		with tf.Session(graph=graph) as session:
			#load the final neural net with priors
			nnet['global_saver'].restore(session, savedir + '/final-prio')
		
			#get the state prior out of the graph (division broadcasting not possible in tf)
			prior = nnet['state_prior'].eval()
		
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
				feed_dict = {nnet['data_in'] : splice(utt_mat,int(self.conf['context_width']))}
				#compute predictions
				p = session.run(predictions, feed_dict=feed_dict)
				#apply prior to predictions
				p = np.divide(p,prior)
				#compute pseudo-likelihood by normalising the weighted predictions
				p = np.divide(p,np.sum(p))
				#write the pseudo-likelihoods in kaldi feature format
				writer.write_next_utt(decodedir + '/feats.ark', utt_id, p)


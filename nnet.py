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
# labels: reference alignments
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)))

#splice the utterance
# utt: utterance to be spliced
# context width: how many franes to the left and right should be concatenated
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
	
def apply_cmvn(utt, stats):
	#compute mean
	mean = stats[0,0:stats.shape[1]-1]/stats[0,stats.shape[1]-1]
	#compute variance
	variance = stats[1,0:stats.shape[1]-1]/stats[0,stats.shape[1]-1] - np.square(mean)
	return np.divide(np.subtract(utt, mean), np.sqrt(variance))

#create a batch of data
def create_batch(reader, reader_cmvn, alignments, utt2spk, input_dim, context_width, num_labels, batch_size, log):
	#create empty batch for input features
	batch_data = np.empty([0,input_dim*(1+2*context_width)], dtype=np.float32)
	#create empty batch for labels
	batch_labels = np.empty([0,0], dtype=np.float32)
	#initialise numbe of utterance in the batch
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

class nnet:
	#ceate class with conf as configuration
	def __init__(self, conf):
		self.conf = conf

	#this function defines the computational graph and does operations on it the input is passed through the dictionary dictin, operations are:
	# init: initialise the neural net no input required
	#	train: train the neural net. input: featdir (directory where feats_shuffled.scp is situated), alignments (dictionary with utt:allignment)
	# prior: compute the state prior, needed to compute pseudo likelihoods. input: featdir (directory where feats_shuffled.scp is situated)
	# decode: compute the state pseudo-likelihood of the input data and write it to kaldi format. input: featdir (directory where feats.scp is situated)
	def graphop(self, op, dictin):
		
		#check that the operation is known
		assert(op in ['init', 'train', 'prior', 'decode'])
					
		#clear the summaries
		if os.path.isdir(self.conf['savedir'] + '/summaries-' + op):
			shutil.rmtree(self.conf['savedir'] + '/summaries-' + op)
		
		# ---------- graph definition ------------		
		graph = tf.Graph()
		with graph.as_default():
			# input
			#input data
			data_in = tf.placeholder(tf.float32, shape = [None, self.conf['input_dim']*(1+2*int(self.conf['context_width']))])
			#output labels
			labels = tf.placeholder(tf.float32)
			
			# variables
			#define weights, biases and their derivatives lists
			weights = []
			dweights = []
			biases = []
			dbiases = []
			
			#input layer, initialise as random normal
			weights.append(tf.Variable(tf.random_normal([self.conf['input_dim']*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])], stddev=float(self.conf['weights_std'])), name = 'W0'))
			dweights.append(tf.Variable(tf.zeros([self.conf['input_dim']*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])]), name = 'dW0'))
			biases.append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units'])], stddev=float(self.conf['biases_std'])), name = 'b0'))
			dbiases.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units'])]), name = 'db0'))
			
			#hidden layers, initialise as random normal
			for i in range(int(self.conf['num_hidden_layers'])-1):
				weights.append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])], stddev=float(self.conf['weights_std'])), name = 'W%d' % (i+1)))
				dweights.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])]), name = 'dW%d' % (i+1)))
				biases.append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units'])], stddev=float(self.conf['biases_std'])), name = 'b%d' % (i+1)))
				dbiases.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units'])]), name = 'db%d' % (i+1)))
			
			#output layer, initialise as zero
			weights.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), self.conf['num_labels']]), name = 'W%d' % int(self.conf['num_hidden_layers'])))
			dweights.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), self.conf['num_labels']]), name = 'dW%d' % int(self.conf['num_hidden_layers'])))
			biases.append(tf.Variable(tf.zeros([self.conf['num_labels']]), name = 'b%d' % int(self.conf['num_hidden_layers'])))
			dbiases.append(tf.Variable(tf.zeros([self.conf['num_labels']]), name = 'db%d' % int(self.conf['num_hidden_layers'])))
			
			#the amount of steps already taken
			global_step = tf.Variable(0, trainable=False, name = 'global_step')
			#the state prior probabilities
			state_prior = tf.Variable(tf.ones([self.conf['num_labels']]), trainable=False, name = 'priors')
			#the total number of steps to be taken
			num_steps = tf.Variable(0, trainable = False, name = 'num_steps')
			#a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
			learning_rate_fact = tf.Variable(tf.ones([], dtype=tf.float32), trainable = False, name = 'learning_rate_fact')
			#the number of frames presented to compute the gradient
			num_frames = tf.Variable(tf.zeros([], dtype=tf.float32), trainable = False, name = 'num_frames')
			#the total loss of the batch
			batch_loss = tf.Variable(tf.zeros([], dtype=tf.float32), trainable = False, name = 'batch_loss')
			
			#propagate does the computations of a single layer 
			#	data: input data to the layer
			#	w: weight matrix
			#	b: bias vector
			#	dropout: flag if dropout should be applied (only for training)
			def propagate(data, w, b, dropout):
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
				if float(self.conf['dropout']) < 1 and dropout:
					data = tf.nn.dropout(data, float(self.conf['dropout']))
				#apply l2 normalisation
				if self.conf['l2_norm'] == 'True':
					data = tf.nn.l2_normalize(data,1)*np.sqrt(float(self.conf['num_hidden_units']))				
				return data

			#model propagates the data through the entire neural net
			def model(data, num_layers, dropout):	

				#propagate through the neural net
				for i in range(num_layers):
					data = propagate(data, weights[i], biases[i], dropout)
					
				#output layer (no softmax yet)
				return tf.matmul(data, weights[int(self.conf['num_hidden_layers'])]) + biases[int(self.conf['num_hidden_layers'])]
				
			#calculate the learning rate, for training use exponential decay. Keep the learning rate fixed for initilialisation
			if op == 'train':
				learning_rate = tf.mul(tf.train.exponential_decay(float(self.conf['initial_learning_rate']), global_step, num_steps, float(self.conf['learning_rate_decay'])), learning_rate_fact)
			else:
				learning_rate = float(self.conf['learning_rate_init'])
			#define the optimizer, with or without momentum
			if float(self.conf['momentum']) > 0:
				optimizer = tf.train.MomentumOptimizer(learning_rate,float(self.conf['momentum']))
			else:
				optimizer = tf.train.GradientDescentOptimizer(learning_rate)
				
			if op == 'init':
				#define the initialisation computation for all the number of layers (needed for layer by layer initialisation). In the initialisation we start with a neural net with one hidden layer, we train the 					hidden layer and the softmax for a couple of steps. We then add a new hidden layer and reinitialise the softmax layer. We then train the added layer and the softmax. We do this until the correct 					number of hidden layers is reached 
				loss = []
				init_optimize = []
				update_gradients_init = []
				for num_layers in range(int(self.conf['num_hidden_layers'])):
			
					#compute the logits (output before softmax)
					logits = model(data_in, num_layers+1, True)
				
					#apply softmax and compute loss
					loss.append(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))/num_frames)
					
					#compute the gradients
					gradients = tf.gradients(loss[num_layers], [weights[num_layers], biases[num_layers], weights[len(weights)-1], biases[len(biases)-1]])
					
					#operations to accumulate the gradients
					update_gradients_init.append([dweights[num_layers].assign(tf.add(dweights[num_layers], gradients[0])).op])
					update_gradients_init[num_layers].append(dbiases[num_layers].assign(tf.add(dbiases[num_layers], gradients[1])).op)
					update_gradients_init[num_layers].append(dweights[len(weights)-1].assign(tf.add(dweights[len(weights)-1], gradients[2])).op)
					update_gradients_init[num_layers].append(dbiases[len(biases)-1].assign(tf.add(dbiases[len(biases)-1], gradients[3])).op)
					
					#operation to accumulate the loss
					update_gradients_init[num_layers].append(batch_loss.assign(tf.add(batch_loss, loss[num_layers])).op)
				
					#list of gradients that will be used to update the parameters
					gradients_to_apply = [(dweights[num_layers].value(), weights[num_layers]), (dweights[len(weights)-1].value(), weights[len(weights)-1]), (dbiases[num_layers].value(), biases[num_layers]), (dbiases[len(biases)-1].value(), biases[len(biases)-1])]
				
					#operation to apply the computed gradients
					init_optimize.append(optimizer.apply_gradients(gradients_to_apply, global_step=global_step))
			else:	
				#define the training computation (forward prop, back prop, update gradients, update params) 
				#compute the logits (output before softmax)
				logits = model(data_in, int(self.conf['num_hidden_layers']), True)
				
				#apply softmax and compute loss
				loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))/num_frames
				
				#do backprop to compute gradients
				gradients = tf.gradients(loss, weights + biases)	
				
				#accumulate the gradients and make a list of gradients that need to be applied to update the parameters
				gradients_to_apply = []
				update_gradients = [batch_loss.assign(tf.add(batch_loss, loss)).op]
				for i in range(len(weights)):
					gradients_to_apply.append((dweights[i].value(), weights[i]))
					gradients_to_apply.append((dbiases[i].value(), biases[i]))
					update_gradients.append(dweights[i].assign(tf.add(dweights[i], gradients[i])).op)
					update_gradients.append(dbiases[i].assign(tf.add(dbiases[i], gradients[len(weights)+i])).op)
					
				#apply the gradients to update the parameters
				optimize = optimizer.apply_gradients(gradients_to_apply, global_step=global_step)
				
			#operation to evaluate the validation set
			logits = model(data_in, int(self.conf['num_hidden_layers']), False)
			loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))/num_frames
			update_val_loss = batch_loss.assign(tf.add(batch_loss, loss)).op
				
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
				weight_summaries.append(tf.histogram_summary('W%d' % i, weights[i]))
				dweight_summaries.append(tf.histogram_summary('dW%d' % i, dweights[i]))
				bias_summaries.append(tf.histogram_summary('b%d' % i, biases[i]))
				dbias_summaries.append(tf.histogram_summary('db%d' % i, dbiases[i]))
		
			#merge summaries
			merged_summary = tf.merge_summary(weight_summaries + dweight_summaries + bias_summaries + dbias_summaries + [loss_summary])
			#define writers
			summary_writer = tf.train.SummaryWriter(self.conf['savedir'] + '/summaries-' + op)
			
			#saver object
			saver = tf.train.Saver(max_to_keep=int(self.conf['check_buffer']))
			
			#if we use the validation set to adapt the learning rate, create a saver to checkpoint the last time the validation set was evaluated
			if self.conf['valid_adapt'] == 'True':
				val_saver = tf.train.Saver(max_to_keep=1)
				
				
		# ---------- operation preprocessing ------------
		if op == 'init':
			#open feature reader
			reader = kaldi_io.KaldiReadIn(dictin['featdir'] + '/feats_shuffled.scp')
			#open the cmvn statistics reader
			reader_cmvn = kaldi_io.KaldiReadIn(dictin['featdir'] + '/cmvn.scp')	
	
			#open log
			log = open(self.conf['savedir'] + '/init.log', 'w')
			
		elif op == 'train':
			#open log
			log = open(self.conf['savedir'] + '/train.log', 'w')
		
			#open feature reader for validation data
			reader = kaldi_io.KaldiReadIn(dictin['featdir'] + '/feats_validation.scp')
			#open cmvn statistics reader
			reader_cmvn = kaldi_io.KaldiReadIn(dictin['featdir'] + '/cmvn.scp')
		
			#create validation set go through all the utterances in feats_validation.scp
			val_data = np.empty([0,self.conf['input_dim']*(1+2*int(self.conf['context_width']))], dtype=np.float32)
			val_labels = np.empty([0,0], dtype=np.float32)
			(utt_id, utt_mat, looped) = reader.read_next_utt()
			while not looped:
				if utt_id in dictin['alignments']:
					#read cmvn stats
					stats = reader_cmvn.read_utt(dictin['utt2spk'][utt_id])
					#apply cmvn
					utt_mat = apply_cmvn(utt_mat, stats)
					
					#add the spliced utterance to batch			
					val_data = np.append(val_data, splice(utt_mat,int(self.conf['context_width'])), axis=0)			
					
					#add labels to batch
					val_labels = np.append(val_labels, dictin['alignments'][utt_id])					
				else:
					log.write('WARNING no alignment for %s, validation set will be smaller\n' % utt_id)
					
				(utt_id, utt_mat, looped) = reader.read_next_utt()

			#put labels in one hot encoding	
			val_labels = (np.arange(self.conf['num_labels']) == val_labels[:,None]).astype(np.float32)	
				
			#open feature reader for training data
			reader = kaldi_io.KaldiReadIn(dictin['featdir'] + '/feats_shuffled.scp')
						
			#go to the initial line (start at point after initilaization and #steps allready taken)
			num_utt = 0
			while num_utt < int(self.conf['batch_size'])*(int(self.conf['starting_step'])+int(self.conf['init_steps'])*int(self.conf['num_hidden_layers'])):
				utt_id = reader.read_next_scp()
				if utt_id in dictin['alignments']:
					num_utt = num_utt + 1
			
		elif op == 'prior':
			#open feature reader
			reader = kaldi_io.KaldiReadIn(dictin['featdir'] + '/feats_shuffled.scp')
			#open cmvn statistics reader
			reader_cmvn = kaldi_io.KaldiReadIn(dictin['featdir'] + '/cmvn.scp')
			
		elif op == 'decode':
			#open feature reader
			reader = kaldi_io.KaldiReadIn(dictin['featdir'] + '/feats.scp')
			#open cmvn statistics reader
			reader_cmvn = kaldi_io.KaldiReadIn(dictin['featdir'] + '/cmvn.scp')
			
			#open prior writer
			writer = kaldi_io.KaldiWriteOut(self.conf['decodedir'] + '/feats.scp')
	
		# ---------- execute operation ------------
		
		#start tensorflow session
		with tf.Session(graph=graph) as session:
		
			if op == 'init':
				
				#initialize the variables
				tf.initialize_all_variables().run()
				
				if self.conf['visualise']=='True':
					summary_writer.add_graph(session.graph_def)
				
				#set the number of steps
				session.run(num_steps.assign(int(self.conf['init_steps'])))
				
				#do layer by layer initialization
				for num_layers in range(int(self.conf['num_hidden_layers'])):
				
					#reinitialize the softmax
					tf.initialize_variables([weights[int(self.conf['num_hidden_layers'])], biases[int(self.conf['num_hidden_layers'])]]).run()
					
					for step in range(int(self.conf['init_steps'])):
						
						#create a batch 
						(batch_data, batch_labels) = create_batch(reader, reader_cmvn, dictin['alignments'], dictin['utt2spk'], self.conf['input_dim'], int(self.conf['context_width']), self.conf['num_labels'], int(self.conf['batch_size']), log)
						
						#tell the neural net how many frames are in the entire batch
						nframes = batch_data.shape[0]
						session.run(num_frames.assign(nframes))
						
						#feed the batch in a number of minibatches to the neural net and accumulate the gradients and loss
						finished = False
						while not finished:
				
							#prepare nnet data
							if batch_data.shape[0] > int(self.conf['mini_batch_size']) and self.conf['mini_batch_size'] != '-1':
								feed_dict = {data_in : batch_data[0:int(self.conf['mini_batch_size']),:], labels : batch_labels[0:int(self.conf['mini_batch_size']),:]}
								batch_data = batch_data[int(self.conf['mini_batch_size']):batch_data.shape[0],:]
								batch_labels = batch_labels[int(self.conf['mini_batch_size']):batch_labels.shape[0],:]
							else:
								feed_dict = {data_in : batch_data, labels : batch_labels}
								finished = True
									
							#do forward backward pass and update gradients					
							session.run(update_gradients_init[num_layers], feed_dict=feed_dict)
							
						#write the summaries to disk so Tensorboard can read them 
						if self.conf['visualise'] == 'True':
							summary_writer.add_summary(merged_summary.eval(), global_step = step + int(self.conf['init_steps'])*num_layers)
						
						#do the appropriate optimization operation
						session.run(init_optimize[num_layers])	
						print("initialization step %d/%d, #layers %d: training loss = %f" % (step + 1, int(self.conf['init_steps']), num_layers+1, batch_loss.eval()))
						
						#reinitlialize the gradients, loss and prediction accuracy
						tf.initialize_variables(dweights + dbiases + [batch_loss]).run()
					
					#set global_step back to 0
					session.run(global_step.assign(0))		
				
				#save the neural net
				saver.save(session, self.conf['savedir'] + '/model', global_step=0)
				
				#put the starting step as 0 so the training op will read the inititial model
				self.conf['starting_step'] = str(0)
				
				#close the log
				log.close()
				
				#close the summary writer so all the summaries still in the pipe are written to disk
				summary_writer.close()
				
			elif op == 'train':
				#load the initial neural net
				saver.restore(session, self.conf['savedir'] + '/model-' + self.conf['starting_step'])
				
				#visualize the graph
				if self.conf['visualise']=='True':
					summary_writer.add_graph(session.graph_def)
				
				#calculate number of steps
				nsteps =  int(int(self.conf['num_epochs']) * len(dictin['alignments']) / int(self.conf['batch_size']))
				
				#set the number of steps
				session.run(num_steps.assign(nsteps))
				
				if self.conf['valid_adapt'] == 'True':
					#initialise old loss to infinity
					old_loss = float('inf')
					#initialise the number of retries (number of consecutive times the training had to go back with half learning rate)
					retry_count = 0
				
				#loop over number of steps
				step = int(self.conf['starting_step'])
				while step < nsteps:
				
					#check performance on evaluation set
					if val_data.shape[0] > 0 and step % int(self.conf['valid_frequency']) == 0:
						#renitialise accuracy 	
						p = 0
						
						#tell the neural net how many frames are in the entire batch
						nframes = val_data.shape[0]
						session.run(num_frames.assign(nframes))
						
						#feed the batch in a number of minibatches to the neural net and accumulate the loss
						i = 0
						while i < nframes:
							#prepare the data
							if self.conf['mini_batch_size'] == '-1':
								end_point = nframes
							else:
								end_point = min(i+int(self.conf['mini_batch_size']), nframes)
								
							feed_dict = {data_in : val_data[i:end_point,:], labels : val_labels[i:end_point,:]}
						
							#accumulate loss and get predictions
							pl, _ = session.run([predictions, update_val_loss], feed_dict = feed_dict)
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
							val_saver.save(session, self.conf['savedir'] + '/validation/validation-checkpoint')
							#update old loss
							old_loss = l_val
							#set retry count to 0
							retry_count = 0
						else:
							#go back to the point where the validation set was previously evaluated
							val_saver.restore(session, self.conf['savedir'] + '/validation/validation-checkpoint')
							
							#if the maximum number of retries has been reached, terminate training
							if retry_count == int(self.conf['valid_retries']):
								print('WARNING: terminating learning (early stopping)')
								break
								
							print('performance on validation set is worse, retrying with halved learning rate')
							#half the learning rate
							session.run(learning_rate_fact.assign(learning_rate_fact.eval()/2))
							#set the step back to the previous point 
							step = step - int(self.conf['valid_frequency'])
							
							#go back in the dataset to the previous point
							num_utt = 0
							while num_utt < int(self.conf['batch_size'])*int(self.conf['valid_frequency']):
								utt_id = reader.read_previous_scp()
								if utt_id in dictin['alignments']:
									num_utt = num_utt + 1
							
							#increment the retry count
							retry_count += 1
					
					#create a training batch 
					(batch_data, batch_labels) = create_batch(reader, reader_cmvn, dictin['alignments'], dictin['utt2spk'], self.conf['input_dim'], int(self.conf['context_width']), self.conf['num_labels'], int(self.conf['batch_size']), log)
					
					#tell the neural net how many frames are in the entire batch
					nframes = batch_data.shape[0]
					session.run(num_frames.assign(nframes))
					
					#feed the batch in a number of minibatches to the neural net and accumulate the gradients and loss (we do it this way to limit memory usage)	
					finished = False
					p = 0
					while not finished:
						
						# prepare data
						if batch_data.shape[0] > int(self.conf['mini_batch_size']) and self.conf['mini_batch_size'] != '-1':
							feed_labels = batch_labels[0:int(self.conf['mini_batch_size']),:]
							feed_dict = {data_in : batch_data[0:int(self.conf['mini_batch_size']),:], labels : feed_labels}
							batch_data = batch_data[int(self.conf['mini_batch_size']):batch_data.shape[0],:]
							batch_labels = batch_labels[int(self.conf['mini_batch_size']):batch_labels.shape[0],:]
						else:
							feed_labels = batch_labels
							feed_dict = {data_in : batch_data, labels : feed_labels}
							finished = True
								
						#do forward-backward pass and update gradients	
						out = session.run([predictions] + update_gradients, feed_dict=feed_dict)
						#update batch accuracy
						p += accuracy(out[0], feed_labels)
					
					#write summaries to disk so Tensorboard can read them
					if self.conf['visualise'] == 'True':
						summary_writer.add_summary(merged_summary.eval(), global_step = step)
					
					#do optimization operation (apply the accumulated gradients)	
					session.run(optimize)			
					print("step %d/%d: training loss = %f, accuracy = %.1f%%, learning rate = %f, #frames in batch = %d" % (step + 1, nsteps, batch_loss.eval(), p/nframes, learning_rate.eval(), nframes))
					
					#reinitlialize the gradients, loss and prediction accuracy
					tf.initialize_variables(dweights + dbiases + [batch_loss]).run()
					
					#save the neural net if at checkpoint
					if step % int(self.conf['check_freq']) == 0:
						saver.save(session, self.conf['savedir'] + '/model', global_step=step)
						
					#increment the step
					step += 1
							
				#save the final neural net
				saver.save(session, self.conf['savedir'] + '/final')
			
				#close the log 
				log .close()
			elif op == 'prior':
				#load the final neural net
				saver.restore(session, self.conf['savedir'] + '/final')
				
				#create the batch to compute the prior
				batch_data = np.empty([0,self.conf['input_dim']*(1+2*int(self.conf['context_width']))], dtype=np.float32)
				num_utt = 0
				for _ in range(int(self.conf['ex_prio'])):
					#read utterance
					(utt_id, utt_mat, looped) = reader.read_next_utt()
					#read cmvn stats
					reader_cmvn.read_utt(dictin['utt2spk'][utt_id])
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
					if batch_data.shape[0] > int(self.conf['mini_batch_size']) and self.conf['mini_batch_size'] != '-1':
						feed_labels = batch_labels[0:int(self.conf['mini_batch_size']),:]
						feed_dict = {data_in : batch_data[0:int(self.conf['mini_batch_size']),:], labels : feed_labels}
						batch_data = batch_data[int(self.conf['mini_batch_size']):batch_data.shape[0],:]
						batch_labels = batch_labels[int(self.conf['mini_batch_size']):batch_labels.shape[0],:]
					else:
						feed_labels = batch_labels
						feed_dict = {data_in : batch_data, labels : feed_labels}
						finished = True
			
					#compute the predictions
					p = session.run(predictions, feed_dict=feed_dict)

					#accumulate the predictions in the prior
					prior += np.sum(p,0)
				
				#normalise the prior
				prior = np.divide(prior, np.sum(prior))
				
				#set the prior in the neural net
				session.run(state_prior.assign(prior))
				
				#save the final neural net with priors
				saver.save(session, self.conf['savedir'] + '/final-prio')
				
				#close the summary writer
				summary_writer.close()
				
			elif op == 'decode':
				#load the final neural net with priors
				saver.restore(session, self.conf['savedir'] + '/final-prio')
				
				#get the state prior out of the graph (division broadcasting not possible in tf)
				prior = state_prior.eval()
				
				#feed the utterances one by one to the neural net
				while True:
					(utt_id, utt_mat, looped) = reader.read_next_utt()
					if looped:
						break
						
					#read the cmvn stats
					stats = reader_cmvn.read_utt(dictin['utt2spk'][utt_id])
					#apply cmvn
					utt_mat = apply_cmvn(utt_mat, stats)
						
					#prepare data
					feed_dict = {data_in : splice(utt_mat,int(self.conf['context_width']))}
					#compute predictions
					p = session.run(predictions, feed_dict=feed_dict)
					#apply prior to predictions
					p = np.divide(p,prior)
					#compute pseudo-likelihood by normalising the weighted predictions
					p = np.divide(p,np.sum(p))
					#write the pseudo-likelihoods in kaldi feature format
					writer.write_next_utt(self.conf['decodedir'] + '/feats.ark', utt_id, p)
			


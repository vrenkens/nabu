import kaldi_io

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import gzip
import shutil
import os

import pdb
import matplotlib.pyplot as plt

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

#splice the utterance
def splice(utt, context_width):
	utt_spliced = np.zeros(shape = [utt.shape[0],utt.shape[1]*(1+2*context_width)], dtype=np.float32)
	utt_spliced[:,context_width*utt.shape[1]:(context_width+1)*utt.shape[1]] = utt
	for i in range(context_width):
		utt_spliced[i+1:utt_spliced.shape[0], (context_width-i-1)*utt.shape[1]:(context_width-i)*utt.shape[1]] = utt[0:utt.shape[0]-i-1,:] #add left context
		utt_spliced[0:utt_spliced.shape[0]-i-1, (context_width+i+1)*utt.shape[1]:(context_width+i+2)*utt.shape[1]] = utt[i+1:utt.shape[0],:] #add right context	
	
	return utt_spliced

#create a batch of data
def create_minibatch(reader, alignments, input_dim, context_width, num_labels, batch_size, log):

	batch_data = np.empty([0,input_dim*(1+2*context_width)], dtype=np.float32)
	batch_labels = np.empty([0,0], dtype=np.float32)
	num_utt = 0
	while num_utt < batch_size:
		(utt_id, utt_mat, _) = reader.read_next_utt()
		if utt_id in alignments:
			#add the spliced utterance to batch			
			batch_data = np.append(batch_data, splice(utt_mat,context_width), axis=0)
			
			#add labels to batch
			batch_labels = np.append(batch_labels, alignments[utt_id])
			
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

	#this function defines the coputational graph and does operations on it the input is passed through the dictionary dictin, operations are:
	# init: initialise the neural net no input required
	#	train: train the neural net. input: featdir (directory where feats_shuffled.scp is situated), alignments (dictionary with utt:allignment)
	# prior: compute the state prior, needed to compute pseudo likelihoods. input: featdir (directory where feats_shuffled.scp is situated)
	# decode: compute the state pseudo-likelihood of the input data and write it to kaldi format. input: featdir (directory where feats.scp is situated)
	def graphop(self, op, dictin):
	
		assert(op in ['init', 'train', 'prior', 'decode'])
		
		# ---------- operation preprocessing ------------
		if op == 'init':
			#open feature reader
			reader = kaldi_io.KaldiReadIn(dictin['featdir'] + '/feats_shuffled.scp')	
			
			#clear the summaries
			if os.path.isdir(self.conf['savedir'] + '/summaries'):
				shutil.rmtree(self.conf['savedir'] + '/summaries')
	
			#open log
			log = open(self.conf['savedir'] + '/init.log', 'w')
			
		elif op == 'train':
			log = open(self.conf['savedir'] + '/train.log', 'w')
		
			reader = kaldi_io.KaldiReadIn(dictin['featdir'] + '/feats_validation.scp')
		
			#create validation set
			val_data = np.empty([0,self.conf['input_dim']*(1+2*int(self.conf['context_width']))], dtype=np.float32)
			val_labels = np.empty([0,0], dtype=np.float32)
			(utt_id, utt_mat, looped) = reader.read_next_utt()
			while not looped:
				if utt_id in dictin['alignments']:
					#add the spliced utterance to batch			
					val_data = np.append(val_data, splice(utt_mat,int(self.conf['context_width'])), axis=0)
				
					#add labels to batch
					val_labels = np.append(val_labels, dictin['alignments'][utt_id])				
				else:
					log.write('WARNING no alignment for %s, validation set will be smaller\n' % utt_id)
					
				(utt_id, utt_mat, looped) = reader.read_next_utt()
				
			#put labels in one hot encoding
			if len(val_labels) > 0:
				val_labels = (np.arange(self.conf['num_labels']) == val_labels[:,None]).astype(np.float32)

			#open feature reader
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
			
		elif op == 'decode':
			#open feature reader
			reader = kaldi_io.KaldiReadIn(dictin['featdir'] + '/feats.scp')
			
			#open prior writer
			writer = kaldi_io.KaldiWriteOut(self.conf['decodedir'] + '/feats.scp')
		
		# ---------- graph definition ------------
		
		graph = tf.Graph()
		with graph.as_default():
			# input
			#input data
			data_in = tf.placeholder(tf.float32, shape = [None, self.conf['input_dim']*(1+2*int(self.conf['context_width']))])
			#output labels
			labels = tf.placeholder(tf.float32)
			if op == 'train' and val_data.shape[0] > 0:
				#validation data
				data_val = tf.constant(val_data)
				#validation labels
				labels_val = tf.constant(val_labels)
			
			# variables
			#define weights and biases in a list
			weights = []
			biases = []
			
			#input layer
			weights.append(tf.Variable(tf.random_normal([self.conf['input_dim']*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])], stddev=float(self.conf['weights_std'])), name = 'W0'))
			biases.append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units'])], stddev=float(self.conf['biases_std'])), name = 'b0'))
			
			#hidden layers
			for i in range(int(self.conf['num_hidden_layers'])-1):
				weights.append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])], stddev=float(self.conf['weights_std'])), name = 'W%d' % (i+1)))
				biases.append(tf.Variable(tf.random_normal([int(self.conf['num_hidden_units'])], stddev=float(self.conf['biases_std'])), name = 'b%d' % (i+1)))
			
			#output layer (zero outut layer)
			weights.append(tf.Variable(tf.zeros([int(self.conf['num_hidden_units']), self.conf['num_labels']]), name = 'W%d' % int(self.conf['num_hidden_layers'])))
			biases.append(tf.Variable(tf.zeros([self.conf['num_labels']]), name = 'b%d' % int(self.conf['num_hidden_layers'])))
			
			#the amount of steps already taken
			global_step = tf.Variable(0, trainable=False, name = 'global_step')
			#the state prior probabilities
			state_prior = tf.Variable(tf.ones([self.conf['num_labels']]), trainable=False, name = 'priors')
			#the total number of steps to be taken
			num_steps = tf.Variable(0, trainable = False, name = 'num_steps')
			#a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
			learning_rate_fact = tf.Variable(tf.ones([], dtype=tf.float32), trainable = False, name = 'learning_rate_fact')
			
			#propagate does the computations of a single layer 
			def propagate(data, w, b):
				data = tf.matmul(data, w) + b
				if self.conf['nonlin'] == 'relu':
					data = tf.maximum(float(self.conf['relu_leak'])*data,data)
				elif self.conf['nonlin'] == 'sigmoid':
					data = tf.nn.sigmoid(data)
				elif self.conf['nonlin'] == 'tanh':
					data = tf.nn.tanh(data)
				else:
					raise Exception('unknown nonlinearity')
					
				if float(self.conf['dropout']) < 1:
					data = tf.nn.dropout(data, float(self.conf['dropout']))
				if self.conf['l2_norm'] == 'True':
					data = tf.nn.l2_normalize(data,1)*np.sqrt(float(self.conf['num_hidden_units']))				
				return data

			#model propagates the data through the entire neural net
			def model(data, num_layers):	

				#propagate through the neural net
				for i in range(num_layers):
					data = propagate(data, weights[i], biases[i])
					
				#output layer (no softmax yet)
				return tf.matmul(data, weights[int(self.conf['num_hidden_layers'])]) + biases[int(self.conf['num_hidden_layers'])]
				
			#calculate the learning rate
			if op == 'train':
				learning_rate = tf.mul(tf.train.exponential_decay(float(self.conf['initial_learning_rate']), global_step, num_steps, float(self.conf['learning_rate_decay'])), learning_rate_fact)
			else:
				learning_rate = float(self.conf['initial_learning_rate'])
			#define the optimizer 
			if float(self.conf['momentum']) > 0:
				optimizer = tf.train.MomentumOptimizer(learning_rate,float(self.conf['momentum']))
			else:
				optimizer = tf.train.GradientDescentOptimizer(learning_rate)
				
			#define the initialisation computation for all the number of layers (needed for layer by layer initialisation). In this optimisation only the final hidden layer and the softmax is updated
			loss = []
			init_optimize = []
			for num_layers in range(int(self.conf['num_hidden_layers'])):
			
				#compute the logits (output before softmax)
				logits = model(data_in, num_layers+1)
				
				#compute the training loss
				loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)))
				
				#optimizing operation
				init_optimize.append(optimizer.minimize(loss[num_layers], global_step=global_step, var_list = [weights[num_layers], biases[num_layers], weights[len(weights)-1], biases[len(biases)-1]]))
			
			#define the training opimisation (gradients are computed first for visualisation purposes)
			if self.conf['visualise_gradients'] == 'True':
				gradients = optimizer.compute_gradients(loss[len(loss)-1])
				optimize = optimizer.apply_gradients(gradients, global_step=global_step)
			else:
				optimize = optimizer.minimize(loss[len(loss)-1], global_step=global_step)
				
			#prediction computation
			predictions = tf.nn.softmax(logits)
			
			#prediction of the validation set
			if op == 'train' and val_data.shape[0] > 0:
				logits_val = model(data_val, int(self.conf['num_hidden_layers']))
				loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_val, labels_val))
				predictions_val = tf.nn.softmax(logits_val)
				
			#create the visualisations							
			if self.conf['visualise_loss'] == 'True':
				#create loss plot
				loss_summary = tf.scalar_summary('loss', loss[len(loss)-1])
			if self.conf['visualise_predictions'] == 'True':
				#create a histogram of predicted labels
				prediction_summary = tf.histogram_summary('prediction', tf.cast(tf.argmax(predictions,1), tf.float32))
			if self.conf['visualise_params'] == 'True':
				#create a histogram of weights and biases
				weight_summaries = []
				bias_summaries = []
				
				for i in range(int(self.conf['num_hidden_layers'])+1):
					weight_summaries.append(tf.histogram_summary('W%d' % i, weights[i]))
					bias_summaries.append(tf.histogram_summary('b%d' % i, biases[i]))
			if self.conf['visualise_gradients'] == 'True':
				#create a histogram of gradients
				gradient_summaries = []
				for gradient in gradients:
					gradient_summaries.append(tf.histogram_summary('d' + gradient[1].name, gradient[0]))
			
			if self.conf['visualise_loss'] == 'True' or self.conf['visualise_predictions'] == 'True' or self.conf['visualise_params'] == 'True' or self.conf['visualise_gradients'] == 'True':
				#merge all summaries
				merged_summary = tf.merge_all_summaries()					
				summary_writer = tf.train.SummaryWriter(self.conf['savedir'] + '/summaries')
			
			#saver object
			saver = tf.train.Saver(max_to_keep=int(self.conf['check_buffer']))
			
			if self.conf['valid_adapt'] == 'True':
				val_saver = tf.train.Saver(max_to_keep=1)
	
		# ---------- execute operation ------------
		
		#start tensorflow session
		with tf.Session(graph=graph) as session:
		
			if op == 'init':
				
				#initialize the variables
				tf.initialize_all_variables().run()
				
				if self.conf['visualise_graph']=='True':
					summary_writer.add_graph(session.graph_def)
				
				#set the number of steps
				session.run(num_steps.assign(int(self.conf['init_steps'])))
				#scale the learning rate
				session.run(learning_rate_fact.assign(int(self.conf['init_lr_scale'])))
				
				#do layer by layer initialization
				for num_layers in range(int(self.conf['num_hidden_layers'])):
				
					#reinitialize the softmax
					tf.initialize_variables([weights[int(self.conf['num_hidden_layers'])], biases[int(self.conf['num_hidden_layers'])]]).run()
					
					for step in range(int(self.conf['init_steps'])):
						#create a minibatch 
						(batch_data, batch_labels) = create_minibatch(reader, dictin['alignments'], self.conf['input_dim'], int(self.conf['context_width']), self.conf['num_labels'], int(self.conf['batch_size']), log)
				
						#prepare nnet data
						feed_dict = {data_in : batch_data, labels : batch_labels}
									
						#do forward backward pass and update					
						if self.conf['visualise_loss'] == 'True' or self.conf['visualise_predictions'] == 'True' or self.conf['visualise_params'] == 'True' or self.conf['visualise_gradients'] == 'True':
							_, l, s = session.run([init_optimize[num_layers], loss[num_layers], merged_summary], feed_dict=feed_dict)
							summary_writer.add_summary(s, global_step = step + int(self.conf['init_steps'])*num_layers)
						else:
							_, l = session.run([init_optimize[num_layers], loss[num_layers]], feed_dict=feed_dict)
							
						print("initialization step %d/%d, #layers %d: training loss = %f" % (step, int(self.conf['init_steps']), num_layers+1, l))
					
					#set global_step back to 0
					session.run(global_step.assign(0))		
				
				#set learning rate factor back to one
				session.run(learning_rate_fact.assign(1))
				
				#save the neural net
				saver.save(session, self.conf['savedir'] + '/model', global_step=0)
				
				self.conf['starting_step'] = str(int(self.conf['starting_step'])+1)
				
				log.close()
				
				summary_writer.close()
				
			elif op == 'train':
			
		
				#load the initial neural net
				saver.restore(session, self.conf['savedir'] + '/model-' + self.conf['starting_step'])
				
				#calculate number of steps
				nsteps =  int(int(self.conf['num_epochs']) * len(dictin['alignments']) / int(self.conf['batch_size']))
				
				#set the number of steps
				session.run(num_steps.assign(nsteps))
				
				if self.conf['valid_adapt'] == 'True':
					old_loss = float('inf')
					retry_count = 0
				
				#loop over number of steps
				step = int(self.conf['starting_step'])
				while step < nsteps:
				
					#create a minibatch
					(batch_data, batch_labels) = create_minibatch(reader, dictin['alignments'], self.conf['input_dim'], int(self.conf['context_width']), self.conf['num_labels'], int(self.conf['batch_size']), log)
				
					#prepare nnet data
					feed_dict = {data_in : batch_data, labels : batch_labels}
									
					#do forward backward pass and update
					if self.conf['visualise_loss'] == 'True' or self.conf['visualise_predictions'] == 'True' or self.conf['visualise_params'] == 'True' or self.conf['visualise_gradients'] == 'True':
						_, l, p, s = session.run([optimize, loss[len(loss)-1], predictions, merged_summary], feed_dict=feed_dict)
						summary_writer.add_summary(s, global_step = step + int(self.conf['init_steps'])*int(self.conf['num_hidden_layers']))
					else:					
						_, l, p = session.run([optimize, loss[len(loss)-1], predictions], feed_dict=feed_dict)
					
					print("step %d/%d: training loss = %f, accuracy = %.1f%%, learning rate = %f" % (step, nsteps, l, accuracy(p, batch_labels), learning_rate.eval()))
					
					#check performance on evaluation set
					if val_data.shape[0] > 0 and step % int(self.conf['valid_frequency']) == 0:
						l_val = loss_val.eval()
						print("validation loss: %f" % (l_val))
						if l_val < old_loss:
							#if performance is better, checkpoint and move on
							val_saver.save(session, self.conf['savedir'] + '/validation/validation-checkpoint')
							old_loss = l_val
							retry_count = 0
						else:
							lrf = learning_rate_fact.eval()
							val_saver.restore(session, self.conf['savedir'] + '/validation/validation-checkpoint')
							
							#if the maximum number of retries has been reached, terminate training
							if retry_count == int(self.conf['valid_retries']):
								print('WARNING: terminating learning (early stopping)')
								break
								
							#if performance is worse half the learning rate and go back to checkpoint
							print('performance on validation set is worse, retrying with halved learning rate')
							session.run(learning_rate_fact.assign(lrf/2))
							step = step - int(self.conf['valid_frequency']) + 1
							retry_count += 1
							continue
					
					#save the neural net if at checkpoint
					if step % int(self.conf['check_freq']) == 0:
						saver.save(session, self.conf['savedir'] + '/model', global_step=step)
						
					step += 1
							
				#save the final neural net
				saver.save(session, self.conf['savedir'] + '/final')
			
			elif op == 'prior':
				#load the final neural net
				saver.restore(session, self.conf['savedir'] + '/final')
				
				#create the batch to compute the prior
				batch_data = np.empty([0,self.conf['input_dim']*(1+2*int(self.conf['context_width']))], dtype=np.float32)
				num_utt = 0
				for _ in range(int(self.conf['ex_prio'])):
					(utt_id, utt_mat, looped) = reader.read_next_utt()
					if looped:
						print('WARNING: not enough utterances to compute the prior')
						break
					
					#add the spiced utterance to batch			
					batch_data = np.append(batch_data, splice(utt_mat,int(self.conf['context_width'])), axis=0)
				
				feed_dict = {data_in : batch_data}
				
				#compute the predictions
				p = session.run(predictions, feed_dict=feed_dict)

				#compute the prior
				prior = np.sum(p,0)
				prior = np.divide(prior, np.sum(prior))
				
				#set the prior
				session.run(state_prior.assign(prior))
				
				#save the final neural net with priors
				saver.save(session, self.conf['savedir'] + '/final-prio')
				
				summary_writer.close()
				
			elif op == 'decode':
				#load the final neural net with priors
				saver.restore(session, self.conf['savedir'] + '/final-prio')
				
				#get the state prior out of the graph (division broadcasting not possible in tf)
				prior = state_prior.eval()
				
				while True:
					(utt_id, utt_mat, looped) = reader.read_next_utt()
					if looped:
						break
						
					#compute the state likelihoods
					feed_dict = {data_in : splice(utt_mat,int(self.conf['context_width']))}
					p = session.run(predictions, feed_dict=feed_dict)
					p = np.divide(p,prior)
					p = np.divide(p,np.sum(p))
					
					
					
					#write the likelihoods in kaldi feature format
					writer.write_next_utt(self.conf['decodedir'] + '/feats.ark', utt_id, p)
					
		if op == 'train':
			log.close()
			


import kaldi_io
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import gzip
import shutil
import os
import pdb

class Nnet:
	#create nnet and define the computational graph
	#	conf: nnet configuration
	#	input_dim: network input dimension
	#	num_labels: number of target labels
	def __init__(self, conf, input_dim, num_labels):
		
		#get nnet structure configs
		self.conf = dict(conf.items('nnet'))
		
		#define location to save neural nets
		self.conf['savedir'] = conf.get('directories','expdir') + '/' + self.conf['name']
		if not os.path.isdir(self.conf['savedir']):
			os.mkdir(self.conf['savedir'])
		if not os.path.isdir(self.conf['savedir'] + '/training'):
			os.mkdir(self.conf['savedir'] + '/training')
		
		#put the number of labels in the config
		self.conf['num_labels'] = num_labels
		
		self.graph = tf.Graph()
		with self.graph.as_default():
			
			#define placeholders
			
			#input data
			data_in = tf.placeholder(tf.float32, shape = [None, input_dim*(1+2*int(self.conf['context_width']))], name = 'input')
			
			#reference labels
			labels = tf.placeholder(tf.float32, shape = [None, num_labels], name = 'targets')
			
			#---define variables---
			
			with tf.variable_scope('train_variables'):	
			
				#total number of steps
				num_steps = tf.get_variable('num_steps', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
			
				#the amount of steps already taken
				global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False) 
			
				#a variable to scale the learning rate (used to reduce the learning rate in case validation performance drops)
				learning_rate_fact = tf.get_variable('learning_rate_fact', [], initializer=tf.constant_initializer(1.0), trainable=False)
			
			#the weights and biases
			weights = [None]*(int(self.conf['num_hidden_layers'])+1)
			biases = [None]*(int(self.conf['num_hidden_layers'])+1)
			
			with tf.variable_scope('model_params'):	
			
				#the state prior probability to compute pseudo-likelihoods
				prior = tf.get_variable('prior', [num_labels], initializer=tf.constant_initializer(1.0), trainable=False)
			
				#input layer
				with tf.variable_scope('layer0'):
					weights[0] = tf.get_variable('weights', [input_dim*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])], initializer=tf.random_normal_initializer(stddev=1/np.sqrt(input_dim)))
					biases[0] = tf.get_variable('biases',  [int(self.conf['num_hidden_units'])], initializer=tf.random_normal_initializer(stddev=float(self.conf['biases_std'])))					

				#hidden layers
				for layer in range(1, int(self.conf['num_hidden_layers'])):
					with tf.variable_scope('layer' + str(layer)):
						weights[layer] = tf.get_variable('weights', [int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])], initializer=tf.random_normal_initializer(stddev=1/np.sqrt(float(self.conf['num_hidden_units']))))
						biases[layer] = tf.get_variable('biases',  [int(self.conf['num_hidden_units'])], initializer=tf.random_normal_initializer(stddev=float(self.conf['biases_std'])))
				
				#output layer
				with tf.variable_scope('layer' + self.conf['num_hidden_layers']):
					weights[-1] = tf.get_variable('weights', [int(self.conf['num_hidden_units']), num_labels], initializer=tf.constant_initializer(0.0))
					biases[-1] = tf.get_variable('biases',  [num_labels], initializer=tf.constant_initializer(0.0))		
			
			with tf.variable_scope('batch_variables'): 	
			
				#number of frames used to compute the gradient
				num_frames = tf.get_variable('num_frames', [], initializer=tf.constant_initializer(0), trainable=False)
			
				#the loss of the complete batch
				batch_loss = tf.get_variable('batch_loss', [], initializer=tf.constant_initializer(0), trainable=False)
						
			#the gradients of the weights and biases
			dweights = [None]*(int(self.conf['num_hidden_layers'])+1)
			dbiases = [None]*(int(self.conf['num_hidden_layers'])+1)
			
			with tf.variable_scope('gradients'):
		
				#input layer
				with tf.variable_scope('layer0'):
					dweights[0] = tf.get_variable('dweights', [input_dim*(1+2*int(self.conf['context_width'])), int(self.conf['num_hidden_units'])], initializer=tf.constant_initializer(0.0), trainable=False)
					dbiases[0] = tf.get_variable('dbiases',  [int(self.conf['num_hidden_units'])], initializer=tf.constant_initializer(0.0), trainable=False)		
		
				#hidden layers
				for layer in range(1, int(self.conf['num_hidden_layers'])):
					with tf.variable_scope('layer' + str(layer)):
						dweights[layer] = tf.get_variable('dweights', [int(self.conf['num_hidden_units']), int(self.conf['num_hidden_units'])], initializer=tf.constant_initializer(0.0), trainable=False)
						dbiases[layer] = tf.get_variable('dbiases',  [int(self.conf['num_hidden_units'])], initializer=tf.constant_initializer(0.0), trainable=False)
		
				#output layer
				with tf.variable_scope('layer' + self.conf['num_hidden_layers']):
					dweights[-1] = tf.get_variable('dweights', [int(self.conf['num_hidden_units']), num_labels], initializer=tf.constant_initializer(0.0), trainable=False)
					dbiases[-1] = tf.get_variable('dbiases',  [num_labels], initializer=tf.constant_initializer(0.0), trainable=False)
						
			#---define the operations that will be used for training---
			
			#create an operation to update the gradients
			#	batchgrads: list of gradients of this minibatch
			#	gradients: variables where the gradients are accumulated in, in the same order as batchgrads
			#	returns: a list of operations to update the gradients
			def update_gradients(batchgrads, gradients):
				operations = [None]*len(gradients)
				for i in range(len(gradients)):
					operations[i] = gradients[i].assign(gradients[i] + batchgrads[i]).op
				
				return operations
				
			#create an operation to apply the gradients
			#	variables: the variables that will be updated
			#	gradients: the gradients of the variables in the same order as variables
			#	optimizer: the optimizer that is being used
			#	global_step: the variable that holds the step (it will be incremented if the gradients are applied)
			#	num_frames: total number of frames in the batch
			#	name: name of the operation
			#	returns: the operation that applies the gradients
			def apply_gradients(variables, gradients, optimizer, global_step, num_frames, name=None):
				gradvar = [None]*len(gradients)
				for i in range(len(gradients)):
					gradvar[i] = (tf.div(gradients[i].value(), num_frames), variables[i])
				
				return optimizer.apply_gradients(gradvar, global_step=global_step, name=name)
				
			#propagate does the computations of a single layer 
			#	data: input data to the layer
			#	w: weight matrix
			#	b: bias vector
			#	dropout: dropout to be applied (should only be used in training)
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
				#apply l2 normalisation
				if self.conf['l2_norm'] == 'True':
					data = tf.nn.l2_normalize(data,1)*np.sqrt(float(self.conf['num_hidden_units']))
				#apply dropout	
				if dropout<1:
					data = tf.nn.dropout(data, dropout)
	
				return data
			
			
			with tf.name_scope('train') as trainscope:
				with tf.name_scope(''):
					with tf.name_scope('decode') as decodescope:
						with tf.name_scope(trainscope):
							#operation to half the learning rate
							tf.group(learning_rate_fact.assign(learning_rate_fact/2).op, name='half_learning_rate')
			
							#compute the learning rate with exponential decay and scale with the learning rate factor
							learning_rate = tf.mul(tf.train.exponential_decay(float(self.conf['initial_learning_rate']), global_step, num_steps, float(self.conf['learning_rate_decay'])), learning_rate_fact, name='learning_rate')
			
							#the optimizer
							optimizer = tf.train.AdamOptimizer(learning_rate)
				
							#the operation to add the number of frames to the number of frames in the batch
							tf.group(num_frames.assign_add(tf.cast(tf.shape(data_in)[0],tf.float32)).op, name='update_nframes')
			
						#set the input data as the data at the current layer
						data = data_in
	
						for layer in range(int(self.conf['num_hidden_layers'])):
							with tf.name_scope(decodescope):
								with tf.variable_scope('layer' + str(layer)):
									#update the data by propagating through the layer
									data = propagate(data, weights[layer], biases[layer], float(self.conf['dropout']))
					
							with tf.name_scope(trainscope):
								with tf.variable_scope('layer' + str(layer)):
									#compute the logits for this layer
									logits = tf.matmul(data, weights[-1]) + biases[-1]
					
									#compute the loss for this layer
									loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
					
									#the layers that can be updated with this loss
									updatelayers =  [-1] + [i for i in xrange(layer+1)]
					
									#comput ethe gradients for the loss at this layer
									gradients = tf.gradients(loss, [weights[i] for i in updatelayers] + [biases[i] for i in updatelayers])
					
									#the operation to add the computed gradients to the batch gradients
									update_grad_ops = tf.group(*update_gradients(gradients, [dweights[i] for i in updatelayers] + [dbiases[i] for i in updatelayers]), name='update_grads')
					
									#the operation to add the computed loss to the batch loss
									tf.group(batch_loss.assign_add(loss).op, name='update_loss')
					
									#operation to apply the gradients and update the parameters
									apply_gradients([weights[i] for i in updatelayers] + [biases[i] for i in updatelayers], [dweights[i] for i in updatelayers] + [dbiases[i] for i in updatelayers], optimizer, global_step, num_frames, 'apply_grads')
			
						with tf.name_scope(trainscope):
							#operation that computes the average loss of the batch
							tf.div(batch_loss, num_frames, name='avrg_loss')
			
						#---define the operation used for testing---
						with tf.name_scope(decodescope):
							with tf.variable_scope('layer' + self.conf['num_hidden_layers']):
								logits = tf.matmul(data, weights[-1]) + biases[-1]
								posterior = tf.nn.softmax(logits)
								
							tf.div(posterior, prior, name='decode')
			
			#--create sumaries for visualisation purposes
			
			if self.conf['visualise'] == 'True':
			
				#loss plot
				summaries = [tf.scalar_summary('loss', batch_loss)]

				#histogram of weights, biases and their gradients
				for i in range(len(weights)):
					summaries.append(tf.histogram_summary('W%d' % i, weights[i]))
					summaries.append(tf.histogram_summary('dW%d' % i, dweights[i]))
					summaries.append(tf.histogram_summary('b%d' % i, biases[i]))
					summaries.append(tf.histogram_summary('db%d' % i, dbiases[i]))

				#merge summaries
				tf.merge_summary(summaries, name='summaries')
		
			
		#create a summary writer for visualisation
		if self.conf['visualise'] == 'True':
			
			#delete the logdir if there is one
			if os.path.isdir(self.conf['savedir'] + '/logdir'):
				shutil.rmtree(self.conf['savedir'] + '/logdir')
			
			self.summarywriter = tf.train.SummaryWriter(logdir=self.conf['savedir'] + '/logdir', graph=self.graph)
	
	# Train the neural network with stochastic gradient descent 
	#	featdir: directory where the features are located
	#	alignments: dictionary containing the state alignments
	#	utt2spk: mapping from utterance to speaker
	def train(self, featdir, alignments, utt2spk):
		
		#open cmvn statistics reader
		reader_cmvn = kaldi_io.KaldiReadIn(featdir + '/cmvn.scp')
		
		#open feature reader for training data
		reader = kaldi_io.KaldiReadIn(featdir + '/feats_shuffled.scp')
		
		if int(self.conf['valid_size']) > 0:
			#create the validation set
			val_data, val_labels = create_batch(reader, reader_cmvn, alignments, utt2spk, int(self.conf['context_width']), self.conf['num_labels'], int(self.conf['valid_size']), int(self.conf['mini_batch_size']))
			
			#split off the utterances that were put into the validation set 
			reader.split()
					
		#go to the point in the database where the training was at checkpoint
		num_utt = 0
		while num_utt < int(self.conf['batch_size'])*int(self.conf['starting_step']):
			utt_id = reader.read_next_scp()
			if utt_id in alignments:
				num_utt = num_utt + 1
		
		#start a tensorflow session
		with tf.Session(graph=self.graph) as session:
			#create the saver
			
			#saver object that saves all the neural net parameters
			saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='model_params') + tf.get_collection(tf.GraphKeys.VARIABLES, scope='train_variables'))
		
			#if the starting step is 0 initialise the neural net, otherwise load a checkpointed one
			step = int(self.conf['starting_step'])
			if step == 0:
				tf.initialize_all_variables().run()
			else:
				saver.restore(session, self.conf['savedir'] + '/training/model-' + str(step))
			
			#compute how many layers have been initialised
			num_layers = min(int(step/int(self.conf['add_layer_period'])), int(self.conf['num_hidden_layers'])-1)
			
			#get the training operations
				
			#update number of frames
			update_nframes = self.graph.get_operation_by_name('train/update_nframes')
			
			#compute the learning rate
			learning_rate = self.graph.get_operation_by_name('train/learning_rate').outputs[0]
			
			#compute the average loss in the batch
			batch_loss = self.graph.get_operation_by_name('train/avrg_loss').outputs[0]
			
			#half the learning rate
			half_learning_rate = self.graph.get_operation_by_name('train/half_learning_rate')
				
			#update the loss
			update_loss = self.graph.get_operation_by_name('train/layer' + str(num_layers) + '/update_loss')
			
			#update the gradients
			update_grads = self.graph.get_operation_by_name('train/layer' + str(num_layers) + '/update_grads')
			
			#apply the gradients
			apply_grads = self.graph.get_operation_by_name('train/layer' + str(num_layers) + '/apply_grads')
			
			#get the placeholders
			data_in = self.graph.get_operation_by_name('input').outputs[0]
			labels = self.graph.get_operation_by_name('targets').outputs[0]
			
			#get the summaries for visualisation
			if self.conf['visualise'] == 'True':
				summaries = self.graph.get_operation_by_name('summaries/summaries').outputs[0]
			
			#calculate number of steps
			nsteps =  int(int(self.conf['num_epochs']) * len(alignments) / int(self.conf['batch_size']))
			
			#set the total number of steps in the graph
			with tf.variable_scope('train_variables', reuse=True):
				tf.get_variable('num_steps', dtype=tf.int32).assign(nsteps).op.run()
		
			retry_count = 0
			old_loss = np.inf
			prev_step = step
		
			#loop over number of steps
			while step < nsteps:
		
				#validation step (before and after a layer is added or at validation frequency with a complete network)
				valid_step = False
				if int(self.conf['valid_size']) > 0:
					if step % int(self.conf['add_layer_period']) == 0 and num_layers <= int(self.conf['num_hidden_layers']):
						valid_step = True
					if step % int(self.conf['valid_frequency']) == 0 and num_layers == int(self.conf['num_hidden_layers']):
						valid_step == True
				
				if valid_step:
					#feed the minibatches to the network one by one and accumulate the loss and number of frames
					for i in range(len(val_data)):
						session.run([update_loss, update_nframes],feed_dict={data_in : val_data[i], labels : val_labels[i]})
						
					#compute the average loss in the validation set
					loss = batch_loss.eval()
					
					print('validation loss: %f' % loss)
					
					#reinitlialize the batch variables (num_frames and loss)
					tf.initialize_variables(tf.get_collection(tf.GraphKeys.VARIABLES, scope='batch_variables')).run()
					
					#check if the loss is better than the previous loss
					if loss < old_loss:
						#set retry count to 0
						retry_count = 0
						
						#save the loss for this validation step
						old_loss = loss
						
						#update the last validation step
						prev_step = step
						
						#save the last validation model
						saver.save(session, self.conf['savedir'] + '/training/model', global_step=step)
						
					#if performance is worse go back to the previous step where the network was validated
					else:
						#if the number of retries exeeds the maximum terminate training
						if retry_count == int(self.conf['valid_retries']):
							print('WARNING: terminating learning (early stopping)')
							break
							
						#increment the retry counter
						retry_count += 1
						
						print('validation is worse, going back to step %d with halved learning rate' % prev_step)
						
						#go back in the dataset to the previous point
						num_utt = 0
						while num_utt < int(self.conf['batch_size'])*(step-prev_step):
							utt_id = reader.read_previous_scp()
							if utt_id in alignments:
								num_utt = num_utt + 1
						
						#load the previous model
						saver.restore(session, self.conf['savedir'] + '/training/model-' + str(prev_step))
						
						#set the step back to the previous step
						step = prev_step
						
						#half the learning rate
						half_learning_rate.run()
				
				#add a layer if needed
				if step % int(self.conf['add_layer_period']) == 0 and step != 0 and num_layers < int(self.conf['num_hidden_layers']) and num_layers*int(self.conf['add_layer_period'])!=step:
					#increment the number of layers
					num_layers += 1
					#reinititalise the output layer
					tf.initialize_variables(tf.get_collection(tf.GraphKeys.VARIABLES, scope='model_params/layer' + self.conf['num_hidden_layers'])).run()
			
					#get the training operations

					#update the loss
					update_loss = self.graph.get_operation_by_name('train/layer' + str(num_layers) + '/update_loss')
			
					#update the gradients
					update_grads = self.graph.get_operation_by_name('train/layer' + str(num_layers) + '/update_grads')
			
					#apply the gradients
					apply_grads = self.graph.get_operation_by_name('train/layer' + str(num_layers) + '/apply_grads')
					
					old_loss = np.inf
					
					continue
			
				
				#training step
			
				#create the training batch in an number of mini_batches
				(batch_data, batch_labels) = create_batch(reader, reader_cmvn, alignments, utt2spk, int(self.conf['context_width']), self.conf['num_labels'], int(self.conf['batch_size']), int(self.conf['mini_batch_size']))

				#feed the minibatches to the network one by one and accumulate the gradients, loss and number of frames
				for i in range(len(batch_data)):
						session.run([update_loss, update_nframes, update_grads],feed_dict={data_in : batch_data[i], labels : batch_labels[i]})
		
				#update the parameters with the accumulated gradients
				(rate, loss, _) = session.run([learning_rate, batch_loss, apply_grads])
		
				print("step %d/%d: training loss = %f, learning rate = %f" % (step + 1, nsteps, loss, rate))
				
				#write a summary of the variables to the log
				if self.conf['visualise'] == 'True':
					self.summarywriter.add_summary(summaries.eval(), global_step=step)

				#reinitlialize all the batch variables (num_frames and loss) and the gradients
				tf.initialize_variables(tf.get_collection(tf.GraphKeys.VARIABLES, scope='batch_variables') + tf.get_collection(tf.GraphKeys.VARIABLES, scope='gradients')).run()
			
				#increment the step
				step += 1
			
				#save the neural net if at checkpoint
				if step % int(self.conf['check_freq']) == 0:
					saver.save(session, self.conf['savedir'] + '/training/model', global_step=step)
			
			#compute the state prior probabilities
			prior = np.array([(np.arange(self.conf['num_labels']) == alignment[:,np.newaxis]).astype(np.float32).sum(0) for alignment in alignments.values()]).sum(1)
			
			#put the prior into the neural net
			with tf.variable_scope('model_params', reuse = True):
				get_variable('prior').assign(prior).op.run()
			
			#save the final neural net
			saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='model_params'))
			saver.save(session, self.conf['savedir'] + '/final')

	#compute pseudo likelihoods for a set of utterances
	#	featdir: directory where the features are located
	#	utt2spk: mapping from utterance to speaker 
	#	decodir: location where output should be stored
	def decode(self, featdir, utt2spk, decodedir):
			
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
		with tf.Session(graph=self.graph) as session:
			
			#load the model
			tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='model_params')).restore(self.conf['savedir'] + '/final')
			
			#get the input placeholder
			data_in = self.graph.get_operation_by_name('input').outputs[0]
			
			#get the output of the neural net
			decode = self.graph.get_operation_by_name('decode/decode').outputs[0]
		
			#feed the utterances one by one to the neural net
			while True:
				(utt_id, utt_mat, looped) = reader.read_next_utt()
				
				if looped:
					break
				
				#read the cmvn stats
				stats = reader_cmvn.read_utt(utt2spk[utt_id])
				
				#apply cmvn
				utt_mat = apply_cmvn(utt_mat, stats)
				
				#compute predictions
				output = decode.eval(feed_dict={data_in : splice(utt_mat,int(self.conf['context_width']))})

				#write the pseudo-likelihoods in kaldi feature format
				writer.write_next_utt(decodedir + '/feats.ark', utt_id, output)
			
		#close the writer
		writer.close()

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
#	context width: number of left and right frames used for splicing
#	num_labels: number of output labels
#	batch_size: size of the batch to be created in number of utterances
#	mini_batch_size: size of the mini_batches
def create_batch(reader, reader_cmvn, alignments, utt2spk, context_width, num_labels, batch_size, mini_batch_size):
	
	num_mini_batches = int((batch_size-1.0)/mini_batch_size) + 1
	batch_data = [None]*num_mini_batches
	batch_labels = [None]*num_mini_batches
	
	for i in range(num_mini_batches):

		#initialise number of utterance in the mini_batch
		if i == num_mini_batches - 1 and batch_size%mini_batch_size !=0:
			num_utt = batch_size%mini_batch_size
		else:
			num_utt = mini_batch_size
		
		n=0
		while n < num_utt:
			#read utterance
			(utt_id, utt_mat, _) = reader.read_next_utt()
			#check if utterance has an alignment
			if utt_id in alignments:
				#read cmvn stats
				stats = reader_cmvn.read_utt(utt2spk[utt_id])
				#apply cmvn
				utt_mat = apply_cmvn(utt_mat, stats)
				
				#add the features and alignments to the mini-batch
				if n == 0:
					batch_data[i] = splice(utt_mat,context_width)
					batch_labels[i] = alignments[utt_id]
				else:
					batch_data[i] = np.append(batch_data[i], splice(utt_mat,context_width), axis=0)	
					batch_labels[i] = np.append(batch_labels[i], alignments[utt_id])
					
				#update number of utterances in the batch
				n += 1
			else:
				print('WARNING no alignment for %s' % utt_id)
		
	#put labels in one hot encoding
	batch_labels = [(np.arange(num_labels) == l[:,np.newaxis]).astype(np.float32) for l in batch_labels]
	
	return (batch_data, batch_labels)

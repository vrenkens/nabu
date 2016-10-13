##@package nnetlayer
# contains neural network layers 

import tensorflow as tf

##This class defines a fully connected feed forward layer
class FFLayer(object):

	##FFLayer constructor, defines the variables
	#
	#@param input_dim input dimension of the layer
	#@param output_dim output dimension of the layer
	#@param weights_std the standart deviation of the weights
	#@param name name of the layer
	#@param activation the activation function used at train time
	#@param trainactivation the activation function used at train time (if not specified no seperate training operations will be created)
	def __init__(self, input_dim, output_dim, weights_std, name, activation, trainactivation = None):
		
		#create the model parameters in this layer
		with tf.variable_scope(name + '_parameters'):
			self.weights = tf.get_variable('weights', [input_dim, output_dim], initializer=tf.random_normal_initializer(stddev=weights_std))
			self.biases = tf.get_variable('biases',  [output_dim], initializer=tf.constant_initializer(0))
				
		#save the parameters
		self.activation = activation
		self.trainactivation = trainactivation
		self.name = name
		
	##Do the forward computation
	#
	#@param inputs the input to the layer
	#@param traininputs the inputs used at train time (if not specified no seperate training operations will be created)
	#
	#@return the output of the layer
	def __call__(self, inputs, traininputs = None):
			
		with tf.name_scope(self.name):
			
			#apply weights and biases
			with tf.name_scope('linear'):
				linear = tf.matmul(inputs, self.weights) + self.biases
				
			#apply activation function	
			with tf.name_scope('activation'):
				outputs = self.activation(linear)
				
			#do the same for the traing inputs if thay have been specified
			if traininputs is not None and self.trainactivation is not None:
				with tf.name_scope('trainlinear'):
					trainlinear = tf.matmul(traininputs, self.weights) + self.biases
					
				with tf.name_scope('trainactivation'):
					trainoutputs = self.trainactivation(trainlinear)
			else:
				trainoutputs = None

		return outputs, trainoutputs
	

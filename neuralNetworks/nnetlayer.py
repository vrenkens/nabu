##@package nnetlayer
# contains neural network layers 

import tensorflow as tf

##This class defines a fully connected feed forward layer
class FFLayer(object):

	##FFLayer constructor, defines the variables
	#
	#@param output_dim output dimension of the layer
	#@param activation the activation function
	#@param weights_std the standart deviation of the weights by default the inverse square root of the input dimension is taken
	#@param trainactivation the activation function used at train time (if not specified no seperate training operations will be created)
	def __init__(self, output_dim, activation, weights_std=None, trainactivation = None):
						
		#save the parameters
		self.output_dim = output_dim
		self.activation = activation
		self.trainactivation = trainactivation
		self.weights_std = weights_std
		
	##Do the forward computation
	#
	#@param inputs the input to the layer
	#@param traininputs the inputs used at train time (if not specified no seperate training operations will be created)
	#@param scope the variable scope of the layer
	#
	#@return the output of the layer and the training output of the layer
	def __call__(self, inputs, traininputs = None, scope = None):
			
		with tf.variable_scope(scope or type(self).__name__):
			with tf.variable_scope('parameters'):
				weights = tf.get_variable('weights', [inputs.get_shape()[1], self.output_dim], initializer=tf.random_normal_initializer(stddev=self.weights_std or 1/int(inputs.get_shape()[1])**0.5))
				biases = tf.get_variable('biases',  [self.output_dim], initializer=tf.constant_initializer(0))
			
			#apply weights and biases
			with tf.variable_scope('linear'):
				linear = tf.matmul(inputs, weights) + biases
				
			#apply activation function	
			with tf.variable_scope('activation'):
				outputs = self.activation(linear)
				
			#do the same for the traing inputs if thay have been specified
			if traininputs is not None and self.trainactivation is not None:
				with tf.name_scope('trainlinear'):
					trainlinear = tf.matmul(traininputs, weights) + biases
					
				with tf.name_scope('trainactivation'):
					trainoutputs = self.trainactivation(trainlinear)
			else:
				trainoutputs = None

		return outputs, trainoutputs
	

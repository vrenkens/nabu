##@package nnetactivations
# contains activation functions

import tensorflow as tf

##a wrapper for an activation function that will add l2 normalisation
class L2_wrapper(object):
	##the L2_wrapper constructor
	#
	#@param activation the activation function being wrapped
	def __init__(self, activation):
		self.activation = activation
		
	##apply the activation function
	#
	#@param inputs the inputs to the activation function
	#
	#@return the output to the activation function
	def __call__(self, inputs):
		
		activations = self.activation(inputs)
		
		with tf.name_scope('l2_norm'):
			#compute the mean squared value
			sig = tf.reduce_mean(tf.square(activations), 1, keep_dims=True)
			
			#divide the input by the mean squared value
			normalized = activations/sig
			
			#if the mean squared value is larger then one select the normalized value otherwise select the unnormalised one
			return tf.select(tf.greater(tf.reshape(sig, [-1]), 1), normalized, activations)
			
## a wrapper for an activation function that will add dropout
class Dropout_wrapper(object):
	##the Dropout_wrapper constructor
	#
	#@param activation the activation function being wrapped
	#@param dopout the dropout rate, has to be a value in (0:1]
	def __init__(self, activation, dropout):
		self.activation = activation
		assert(dropout > 0 and dropout <= 1)
		self.dropout = dropout
		
	##apply the activation function
	#
	#@param inputs the inputs to the activation function
	#
	#@return the output to the activation function
	def __call__(self, inputs):
		activations = self.activation(inputs)
		return tf.nn.dropout(activations, self.dropout)
		
		
			

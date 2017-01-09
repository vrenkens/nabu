'''@file dnn.py
The DNN neural network classifier'''

import seq_convertors
import tensorflow as tf
import activation
from classifier import Classifier
from layer import FFLayer

class DNN(Classifier):
    '''a DNN classifier'''

    def __call__(self, inputs, input_seq_length, targets=None,
                 target_seq_length=None, is_training=False, reuse=False,
                 scope=None):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] dimansional vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length x 1] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] dimansional vector
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the name scope

        Returns:
            A quadruple containing:
                - output logits
                - the output logits sequence lengths as a vector
                - a saver object
                - a dictionary of control operations (may be empty)
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #build the activation function

            #batch normalisation
            if self.conf['batch_norm'] == 'True':
                act = activation.Batchnorm(None)
            else:
                act = None

            #non linearity
            if self.conf['nonlin'] == 'relu':
                act = activation.TfActivation(act, tf.nn.relu)
            elif self.conf['nonlin'] == 'sigmoid':
                act = activation.TfActivation(act, tf.nn.sigmoid)
            elif self.conf['nonlin'] == 'tanh':
                act = activation.TfActivation(act, tf.nn.tanh)
            elif self.conf['nonlin'] == 'linear':
                act = activation.TfActivation(act, lambda(x): x)
            else:
                raise Exception('unkown nonlinearity')

            #L2 normalization
            if self.conf['l2_norm'] == 'True':
                act = activation.L2Norm(act)

            #dropout
            if float(self.conf['dropout']) < 1:
                act = activation.Dropout(act, float(self.conf['dropout']))

            #input layer
            layer = FFLayer(int(self.conf['num_units']), act)

            #output layer
            outlayer = FFLayer(self.output_dim,
                               activation.TfActivation(None, lambda(x): x), 0)

            #do the forward computation

            #convert the sequential data to non sequential data
            nonseq_inputs = seq_convertors.seq2nonseq(inputs, input_seq_length)

            activations = [None]*int(self.conf['num_layers'])
            activations[0] = layer(nonseq_inputs, is_training, reuse, 'layer0')
            for l in range(1, int(self.conf['num_layers'])):
                activations[l] = layer(activations[l-1], is_training, reuse,
                                       'layer' + str(l))

            if self.conf['layerwise_init'] == 'True':

                #variable that determines how many layers are initialised
                #in the neural net
                initialisedlayers = tf.get_variable(
                    'initialisedlayers', [],
                    initializer=tf.constant_initializer(0),
                    trainable=False,
                    dtype=tf.int32)

                #operation to increment the number of layers
                add_layer_op = initialisedlayers.assign(initialisedlayers+1).op

                #compute the logits by selecting the activations at the layer
                #that has last been added to the network, this is used for layer
                #by layer initialisation
                logits = tf.case(
                    [(tf.equal(initialisedlayers, tf.constant(l)),
                      Callable(activations[l]))
                     for l in range(len(activations))],
                    default=Callable(activations[-1]),
                    exclusive=True, name='layerSelector')

                logits.set_shape([None, int(self.conf['num_units'])])
            else:
                logits = activations[-1]

            logits = outlayer(logits, is_training, reuse,
                              'layer' + self.conf['num_layers'])


            if self.conf['layerwise_init'] == 'True':
                #operation to initialise the final layer
                init_last_layer_op = tf.initialize_variables(
                    tf.get_collection(
                        tf.GraphKeys.VARIABLES,
                        scope=(tf.get_variable_scope().name + '/layer'
                               + self.conf['num_layers'])))

                control_ops = {'add':add_layer_op, 'init':init_last_layer_op}
            else:
                control_ops = None

            #convert the logits to sequence logits to match expected output
            seq_logits = seq_convertors.nonseq2seq(logits, input_seq_length,
                                                   int(inputs.get_shape()[1]))

            #create a saver
            saver = tf.train.Saver()

        return seq_logits, input_seq_length, saver, control_ops

class Callable(object):
    '''A class for an object that is callable'''

    def __init__(self, value):
        '''
        Callable constructor

        Args:
            tensor: a tensor
        '''

        self.value = value

    def __call__(self):
        '''
        get the object

        Returns:
            the object
        '''

        return self.value

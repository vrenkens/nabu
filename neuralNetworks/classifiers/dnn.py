'''@file dnn.py
The DNN neural network classifier'''

import tensorflow as tf
from classifier import Classifier
from layer import FFLayer
from activation import TfActivation

class DNN(Classifier):
    '''This class is a graph for feedforward fully connected neural nets.'''

    def __init__(self, output_dim, num_layers, num_units, activation,
                 layerwise_init=True):
        '''
        DNN constructor

        Args:
            output_dim: the DNN output dimension
            num_layers: number of hidden layers
            num_units: number of hidden units
            activation: the activation function
            layerwise_init: if True the layers will be added one by one,
                otherwise all layers will be added to the network in the
                beginning
        '''

        #super constructor
        super(DNN, self).__init__(output_dim)

        #save all the DNN properties
        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = activation
        self.layerwise_init = layerwise_init

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        '''
        Add the DNN variables and operations to the graph

        Args:
            inputs: the inputs to the neural network
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the name scope

        Returns:
            A triple containing:
                - output logits
                - a saver object
                - a dictionary of control operations:
                    -add: add a layer to the network
                    -init: initialise the final layer
        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #input layer
            layer = FFLayer(self.num_units, self.activation)

            #output layer
            outlayer = FFLayer(self.output_dim,
                               TfActivation(None, lambda(x): x), 0)

            #do the forward computation with dropout

            activations = [None]*self.num_layers
            activations[0] = layer(inputs, is_training, reuse, 'layer0')
            for l in range(1, self.num_layers):
                activations[l] = layer(activations[l-1], is_training, reuse,
                                       'layer' + str(l))

            if self.layerwise_init:

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
                      CallableTensor(activations[l]))
                     for l in range(len(activations))],
                    default=CallableTensor(activations[-1]),
                    exclusive=True, name='layerSelector')

                logits.set_shape([None, self.num_units])

            else:
                logits = activations[-1]

            logits = outlayer(logits, is_training, reuse,
                              'layer' + str(self.num_layers))


            if self.layerwise_init:
                #operation to initialise the final layer
                init_last_layer_op = tf.initialize_variables(
                    tf.get_collection(
                        tf.GraphKeys.VARIABLES,
                        scope=(tf.get_variable_scope().name + '/layer'
                               + str(self.num_layers))))

                control_ops = {'add':add_layer_op, 'init':init_last_layer_op}
            else:
                control_ops = None

            #create a saver
            saver = tf.train.Saver()

        return logits, saver, control_ops

class CallableTensor(object):
    '''A class for a tensor that is callable'''

    def __init__(self, tensor):
        '''
        CallableTensor constructor

        Args:
            tensor: a tensor
        '''

        self.tensor = tensor

    def __call__(self):
        '''
        get the tensor

        Returns:
            the tensor
        '''

        return self.tensor

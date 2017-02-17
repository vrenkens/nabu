'''@file dnn.py
The DNN neural network classifier'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import classifier, layer, activation

class DNN(classifier.Classifier):
    '''a DNN classifier'''

    def _get_outputs(self, inputs, input_seq_length, targets=None,
                     target_seq_length=None, is_training=False):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

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

        #input and hidden layer
        hidlayer = layer.Linear(int(self.conf['num_units']))

        #output layer
        outlayer = layer.Linear(self.output_dim)

        #do the forward computation

        activations = [None]*int(self.conf['num_layers'])
        activations[0] = act(hidlayer(inputs, 'layer0'), is_training)
        for l in range(1, int(self.conf['num_layers'])):
            activations[l] = act(hidlayer(activations[l-1],
                                          'layer' + str(l)), is_training)

        logits = activations[-1]

        logits = outlayer(logits, 'layer' + self.conf['num_layers'])

        return logits, input_seq_length

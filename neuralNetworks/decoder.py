'''@file decoder.py
neural network decoder environment'''

import tensorflow as tf

class Decoder(object):
    '''Class for the decoding environment for a neural net classifier'''

    def __init__(self, classifier, input_dim):
        '''
        NnetDecoder constructor, creates the decoding graph

        Args:
            classifier: the classifier that will be used for decoding
            input_dim: the input dimension to the nnnetgraph
        '''

        self.graph = tf.Graph()

        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(tf.float32, shape=[None, input_dim],
                                         name='inputs')

            split_inputs = tf.unpack(tf.reshape(self.inputs,
                                                [-1, 1, input_dim]))

            #create the decoding graph
            logits, self.saver, _ = classifier(split_inputs, None,
                                               is_training=False, reuse=False,
                                               scope='Classifier')

            logits = tf.reshape(tf.pack(logits), [-1, classifier.output_dim])

            #compute the outputs
            self.outputs = tf.nn.softmax(logits)

        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

    def __call__(self, inputs):
        '''decode using the neural net

        Args:
            inputs: the inputs to the graph as a NxF numpy array where N is the
                number of frames and F is the input feature dimension

        Returns:
            an NxO numpy array where N is the number of frames and O is the
                neural net output dimension
        '''

        #pylint: disable=E1101
        return self.outputs.eval(feed_dict={self.inputs:inputs})

    def restore(self, filename):
        '''
        load the saved neural net

        Args:
            filename: location where the neural net is saved
        '''

        self.saver.restore(tf.get_default_session(), filename)

'''@file decoder.py
neural network decoder environment'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np

def decoder_factory(conf,
                    classifier,
                    input_dim,
                    max_input_length,
                    expdir,
                    decoder_type):
    '''
    creates a decoder object

    Args:
        conf: the decoder config
        classifier: the classifier that will be used for decoding
        input_dim: the input dimension to the nnnetgraph
        max_input_length: the maximum length of the inputs
        expdir: the location where the models were saved and the results
            will be written
        decoder_type: the decoder type
    '''

    if decoder_type == 'ctcdecoder':
        decoder_class = CTCDecoder
    else:
        raise Exception('Undefined decoder type: %s' % decoder_type)

    return decoder_class(conf,
                         classifier,
                         input_dim,
                         max_input_length,
                         expdir)

class Decoder(object):
    '''the abstract class for a decoder'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, classifier, input_dim, max_input_length, expdir):
        '''
        Decoder constructor, creates the decoding graph

        Args:
            conf: the decoder config
            classifier: the classifier that will be used for decoding
            input_dim: the input dimension to the nnnetgraph
            max_input_length: the maximum length of the inputs
            expdir: the location where the models were saved and the results
                will be written
        '''

        self.conf = conf
        self.max_input_length = max_input_length
        self.expdir = expdir

        self.graph = tf.Graph()

        with self.graph.as_default():

            #create the inputs placeholder
            self.inputs = tf.placeholder(
                tf.float32, shape=[1, max_input_length, input_dim],
                name='inputs')

            #create the sequence length placeholder
            self.input_seq_length = tf.placeholder(
                tf.int32, shape=[1], name='seq_length')

            #create the decoding graph
            logits, logits_seq_length, self.saver, _ =\
                classifier(
                    self.inputs, self.input_seq_length, targets=None,
                    target_seq_length=None, is_training=False,
                    reuse=False, scope='Classifier')

            #compute the outputs based on the classifier output logits
            self.outputs = self.get_outputs(logits, logits_seq_length)

        #specify that the graph can no longer be modified after this point
        self.graph.finalize()

    @abstractmethod
    def get_outputs(self, logits, logits_seq_length):
        '''
        Compute the outputs based on the output classifier output logits

        Args:
            logits: a NxO tensor where N is the sequence length and O is the
                classifier output dimension
            logits_seq_length: the logits sequence length

        Returns:
            the outputs of the decoding graph
        '''

        raise NotImplementedError("Abstract method")

    @abstractmethod
    def process_decoded(self, decoded):
        '''
        do some postprocessing on the output of the decoding graph

        Args:
            decoded: the outputs of the decoding graph

        Returns:
            a list of pairs containing:
                - the score of the output
                - the output lable sequence as a numpy array
        '''

    def decode(self, reader, coder):
        '''decode using the neural net

        Args:
            reader: a feauture reader object containing the testng features
            coder: a target coder object to write the results to disk

        Returns:
            the output of the decoder
        '''

        #start the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        config.allow_soft_placement = True

        with tf.Session(graph=self.graph, config=config) as sess:

            #load the trained model
            self.saver.restore(sess, self.expdir + '/logdir/final.ckpt')

            while True:

                (utt_id, inputs, looped) = reader.get_utt()

                if looped:
                    break

                #get the sequence length
                input_seq_length = [inputs.shape[0]]

                #pad the inputs
                inputs = np.append(
                    inputs, np.zeros([self.max_input_length-inputs.shape[0],
                                      inputs.shape[1]]), 0)

                #pylint: disable=E1101
                decoded = sess.run(
                    self.outputs,
                    feed_dict={self.inputs:inputs[np.newaxis, :, :],
                               self.input_seq_length:input_seq_length})

                decoded = self.process_decoded(decoded)

                #write the results to disk
                with open(self.expdir + '/decoded/' + utt_id, 'w') as fid:
                    for d in decoded:
                        fid.write('%f\t%s\n' % (d[0], coder.decode(d[1])))

        return decoded

    def restore(self, filename):
        '''
        load the saved neural net

        Args:
            filename: location where the neural net is saved
        '''

        self.saver.restore(tf.get_default_session(), filename)


class CTCDecoder(Decoder):
    '''CTC Decoder'''

    def get_outputs(self, logits, logits_seq_length):
        '''
        get the outputs with ctc beam search

        Args:
            logits: A list containing a 1xO tensor for each timestep where O
                is the classifier output dimension
            logits_seq_length: the logits sequence length

        Returns:
            a tupple of length beam_width + 1 where the first beam_width
            elements are vectors with label sequences and the last elements
            is a beam_width length vector containing scores
        '''

        #Convert logits to time major
        logits = tf.pack(tf.unpack(logits, axis=1))

        #do the CTC beam search
        sparse_outputs, logprobs = tf.nn.ctc_beam_search_decoder(
            tf.pack(logits), logits_seq_length, int(self.conf['beam_width']),
            int(self.conf['beam_width']))

        #convert the outputs to dense tensors
        dense_outputs = [
            tf.reshape(tf.sparse_tensor_to_dense(o), [-1])
            for o in sparse_outputs]

        return dense_outputs + [tf.reshape(logprobs, [-1])]

    def process_decoded(self, decoded):
        '''
        create numpy arrays of decoded targets

        Args:
            decoded: a tupple of length beam_width + 1 where the first
                beam_width elements are vectors with label sequences and the
                last elements is a beam_width length vector containing scores

        Returns:
            a list of pairs containing:
                - the score of the output
                - the output lable sequence as a numpy array
        '''

        target_sequences = decoded[:-1]
        logprobs = decoded[-1]

        processed = [(logprobs[b], target_sequences[b])
                     for b in range(len(target_sequences))]

        return processed

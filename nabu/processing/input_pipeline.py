'''@file input_pipeline.py
contains the methotology for creating the input pipeline'''

from __future__ import division
import os
import numpy as np
import tensorflow as tf
from tfreaders import tfreader_factory

def get_filenames(dataconfs):
    '''create a list of filenames to put into the queue

    Args:
        dataconfs: the database configurations as a list of lists

    Returns:
        - a list containing all the filenames
        - a list containing the names
    '''

    #create the list of strings that will be put into the shared data queue
    #one queue element is a space seperated list of all data elements for
    #one example

    #read all the names and files
    files = []
    for dataconfset in dataconfs:
        setfiles = dict()
        for i, dataconf in enumerate(dataconfset):
            with open(os.path.join(dataconf['dir'], 'pointers.scp')) as fid:
                for line in fid:
                    (n, f) = line.strip().split('\t')
                    setfiles['%s-%d' % (n, i)] = f
        files.append(setfiles)

    #loop over the first names and look for them in the other names. If not
    #all sets contain the name, ignore it
    data_queue_elements = []
    names = []
    for name in files[0]:
        allfound = True
        for setfile in files:
            if name not in setfile:
                print('%s was not found in all sets of data, ignoring this'
                      ' example' % name)
                allfound = False
                break
        if allfound:
            data_queue_element = files[0][name]
            for setfile in files[1:]:
                data_queue_element += '\t' + setfile[name]
            data_queue_elements.append(data_queue_element)
            names.append(name)

    return data_queue_elements, names

def input_pipeline(
    data_queue,
    batch_size,
    numbuckets,
    dataconfs,
    variable_batch_size=False,
    allow_smaller_final_batch=False,
    name=None):
    '''create the input pipeline

    Args:
        data_queue: the data queue where the filenemas are queued
        batch_size: the desired batch size
        numbuckets: the number of data buckets
        dataconfs: the databes configuration sections that should be read
            as a list of lists
        variable_batch_size: bool, change batch size from bucket to bucket,
            for buckets with higher seq_length a smaller batch size is used
        allow_smaller_final_batch: if set to True a smaller final batch is
            allowed
        name: name of the pipeline

    Returns:
        - the data elements as a list of [batch_size x ...] tensor
        - the sequence lengths as a list of [batch_size] tensor
        - the number of steps in each epoch'''

    with tf.variable_scope(name or 'input_pipeline'):

        #split the an element in the data queue and enqueue them
        #in seperaterately in a different queue
        with tf.name_scope('split_queue'):

            filenames = tf.sparse_tensor_to_dense(tf.string_split(
                [data_queue.dequeue()], '\t'), '')
            filenames.set_shape([1, len(dataconfs)])
            filenames = tf.unstack(tf.reshape(filenames, [-1]))

        data = []

        with tf.variable_scope('read_data'):
            #create a seperate queue for each data element
            for i, dataconfset in enumerate(dataconfs):
                with tf.variable_scope('reader'):

                    queue = tf.FIFOQueue(
                        capacity=1,
                        dtypes=[tf.string],
                        shapes=[[]],
                        name='split_queue'
                    )

                    enqueue_op = queue.enqueue(filenames[i])

                    #create a reader to read from the queue
                    types = [dataconf['type'] for dataconf in dataconfset]
                    if len(set(types)) > 1:
                        raise Exception(
                            'all data types in a set must be the same')
                    dirs = [dataconf['dir'] for dataconf in dataconfset]
                    reader = tfreader_factory.factory(types[0])(dirs)

                    if i == 0:
                        sequence_length_histogram = \
                            reader.metadata['sequence_length_histogram']

                    #read the data from the data element queue and make sure
                    #they happen in the correct order
                    with tf.control_dependencies([enqueue_op]):
                        read_data = reader(queue)
                        data += read_data

            data = tf.tuple(data)

        #create batches of the data
        if numbuckets > 1:
            boundaries = bucket_boundaries(sequence_length_histogram,
                                           numbuckets)
            if variable_batch_size:
                batch_sizes = [
                    max(int(batch_size*boundaries[0]/b), 1)
                    for b in boundaries + [sequence_length_histogram.size]]

                #compute the number of steps
                numutt = [
                    sequence_length_histogram[boundaries[i]:b].sum()
                    for i, b in enumerate(boundaries[1:])]
                numutt = (
                    [sequence_length_histogram[:boundaries[0]].sum()] +
                    numutt +
                    [sequence_length_histogram[boundaries[-1]:].sum()])
                num_steps = int((np.array(numutt)/np.array(batch_sizes)).sum())
            else:
                batch_sizes = int(batch_size)
                num_steps = int(sequence_length_histogram.sum()/batch_sizes)
            _, batches = tf.contrib.training.bucket_by_sequence_length(
                input_length=data[1],
                tensors=data,
                batch_size=batch_sizes,
                bucket_boundaries=boundaries,
                allow_smaller_final_batch=allow_smaller_final_batch,
                dynamic_pad=True
            )
        else:
            num_steps = int(sequence_length_histogram.sum()/int(batch_size))
            batches = tf.train.batch(
                tensors=data,
                batch_size=int(batch_size),
                capacity=int(batch_size)*2,
                allow_smaller_final_batch=allow_smaller_final_batch,
                dynamic_pad=True)

        #seperate the data and the sequence lengths
        data = batches[::2]
        seq_length = batches[1::2]

        return data, seq_length, num_steps

def bucket_boundaries(histogram, numbuckets):
    '''detemine the bucket boundaries to uniformally devide the number of
    elements in the buckets


    this is a greedy algorithm and does not guarantee an optimal solution'''

    boundaries = [0]*(numbuckets)
    for i in range(numbuckets-1):
        numelements = int(histogram[boundaries[i]:].sum()/(numbuckets-i))
        if numelements == 0:
            print '%d buckets could not be reached, using %d buckets' % (
                numbuckets, i)

        #add elements to the bucket to get as close as possible to the desired
        #number of elements
        j = boundaries[i] + 1

        while (j+1 < len(histogram) and
               abs(histogram[boundaries[i]:j].sum() - numelements) >=
               abs(histogram[boundaries[i]:j+1].sum() - numelements)):

            j += 1

        boundaries[i+1] = j

    return boundaries[1:]

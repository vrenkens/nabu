'''@file hooks.py
contains session hooks'''

import tensorflow as tf

class LoadAtBegin(tf.train.SessionRunHook):
    '''a training hook for saving the final variables'''

    def __init__(self, filename, variables):
        '''hook constructor

        Args:
            filename: where the model will be saved
            variables: the variables that will be loaded'''

        self.filename = filename
        self.variables = variables

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver(self.variables, sharded=True)

    def after_create_session(self, session, _):
        '''this will be run after session creation'''

        self._saver.restore(session, self.filename)

class SaveAtEnd(tf.train.SessionRunHook):
    '''a training hook for saving the final variables'''

    def __init__(self, filename, variables):
        '''hook constructor

        Args:
            filename: where the model will be saved
            variables: the variables that will be saved'''

        self.filename = filename
        self.variables = variables

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver(self.variables, sharded=True)

    def end(self, session):
        '''this will be run at session closing'''

        self._saver.save(session, self.filename)

class ValidationSaveHook(tf.train.SessionRunHook):
    '''a training hook for saving and loading the validated model'''
    def __init__(self, filename, model):
        '''hook constructor

        Args:
            filename: where the model will be saved
            model: the model that will be saved'''

        self.filename = filename
        self.model = model

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver(sharded=True)

    def after_create_session(self, session, _):
        '''this will be run after session creation'''

        #pylint: disable=W0201
        self._sess = session

    def save(self):
        '''save the current parameters'''

        self._saver.save(self._sess, self.filename)

    def restore(self):
        '''restore the previously validate parameters'''

        self._saver.restore(self._sess, self.filename)


class StopHook(tf.train.SessionRunHook):
    '''a hook that makes sure all replicas terminate when session ends'''

    def __init__(self, done_op):
        '''hook constructor'''

        self.done_op = done_op

    def end(self, session):
        '''this will be run at session closing'''

        self.done_op.run(session=session)

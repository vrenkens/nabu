'''@file hooks.py
contains session hooks'''

import tensorflow as tf

class LoadAtBegin(tf.train.SessionRunHook):
    '''a training hook for saving the final model'''

    def __init__(self, filename, model):
        '''hook constructor

        Args:
            filename: where the model will be saved
            model: the model that will be loaded'''

        self.filename = filename
        self.model = model

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver(self.model.variables, sharded=True)

    def after_create_session(self, session, _):
        '''this will be run after session creation'''

        self._saver.restore(session, self.filename)

class SummaryHook(tf.train.SessionRunHook):
    '''a training hook for saving the final model'''

    def __init__(self, logdir):
        '''hook constructor

        Args:
            filename: where the model will be saved'''

        self.logdir = logdir

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._summary = tf.summary.merge_all()


    def after_create_session(self, session, _):
        '''this will be run after session creation'''

        #pylint: disable=W0201
        self._writer = tf.summary.FileWriter(self.logdir, session.graph)

    def before_run(self, _):
        '''this will be executed before a session run call'''

        return tf.train.SessionRunArgs(fetches={'summary':self._summary})

    def after_run(self, _, run_values):
        '''this will be executed after a run call'''

        self._writer.add_summary(run_values.results['summary'])

class SaveAtEnd(tf.train.SessionRunHook):
    '''a training hook for saving the final model'''

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
        self._saver = tf.train.Saver(self.model.variables, sharded=True)

    def end(self, session):
        '''this will be run at session closing'''

        self._saver.save(session, self.filename)

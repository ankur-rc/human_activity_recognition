'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 9:23:41 pm
Author: ankurrc
'''
from metrics import F1_metrics
from keras.callbacks import TensorBoard


class Model(object):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None, batch_size=128, epochs=25, verbosity=1):
        self.verbose = verbosity
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_timesteps = train_data["X"].shape[1]
        self.n_features = train_data["X"].shape[2]
        self.n_outputs = train_data["y"].shape[1]

        self.train_X = train_data["X"]
        self.train_y = train_data["y"]
        self.test_X = test_data["X"]
        self.test_y = test_data["y"]

        self.callbacks = [F1_metrics(), TensorBoard(
            log_dir=tb_log_dir, write_grads=True, write_graph=True, histogram_freq=3, batch_size=self.batch_size)]

        self.model = None

    def build(self, model):
        self.model = model

    def evaluate(self):
        self.model.fit(self.train_X, self.train_y, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=self.verbose, validation_split=0.2, callbacks=self.callbacks, shuffle=True)

        # evaluate model
        _, accuracy = self.model.evaluate(
            self.test_X, self.test_y, batch_size=self.batch_size, verbose=self.verbose)
        return accuracy

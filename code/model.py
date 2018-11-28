'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 9:23:41 pm
Author: ankurrc
'''


class Model(object):

    def __init__(self, train_data=None, test_data=None):
        self.verbose = 1
        self.epochs = 15
        self.batch_size = 1
        self.n_timesteps = train_data.X.shape[1]
        self.n_features = train_data.X.shape[2]
        self.n_outputs = train_data.y.shape[1]

    def build(self):
        raise NotImplementedError("Use a subclass!")

    def evaluate(self):
        raise NotImplementedError("Use a subclass!")

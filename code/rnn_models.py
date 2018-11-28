'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 9:38:33 pm
Author: ankurrc
'''
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D
from keras.callbacks import TensorBoard

from model import Model
from data import Dataset
from metrics import F1_metrics


class LSTM_model(Model):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data, tb_log_dir=tb_log_dir)
        self.build()

    def evaluate(self, log_dir=None):
        accuracy = super().evaluate()

        return accuracy

    def build(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(
            self.n_timesteps, self.n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        super().build(model)


class CNN_LSTM_model(Model):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data,
                         tb_log_dir=tb_log_dir, batch_size=64)
        self.build()

    def evaluate(self, log_dir=None):
        accuracy = super().evaluate()

        return accuracy

    def build(self):
        n_steps, n_length = 4, 32

        self.train_X = self.train_X.reshape(
            (self.train_X.shape[0], n_steps, n_length, self.n_features))
        self.test_X = self.test_X.reshape(
            (self.test_X.shape[0], n_steps, n_length, self.n_features))

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3,
                                         activation='relu'), input_shape=(None, n_length, self.n_features)))
        model.add(TimeDistributed(
            Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        super().build(model)


class ConvLSTM_model(Model):

    def __init__(self, train_data=None, test_data=None, tb_log_dir=None):
        super().__init__(train_data=train_data, test_data=test_data,
                         tb_log_dir=tb_log_dir, batch_size=64)
        self.build()

    def evaluate(self, log_dir=None):
        accuracy = super().evaluate()

        return accuracy

    def build(self):
        n_steps, n_length = 4, 32

        self.train_X = self.train_X.reshape(
            (self.train_X.shape[0], n_steps, 1, n_length, self.n_features))
        self.test_X = self.test_X.reshape(
            (self.test_X.shape[0], n_steps, 1, n_length, self.n_features))

        # define model
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3),
                             activation='relu', input_shape=(n_steps, 1, n_length, self.n_features)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        super().build(model)


if __name__ == "__main__":
    dataset_root = "/media/ankurrc/new_volume/633_ml/project/code/dataset/UCI HAR Dataset/"
    dataset = Dataset(dataset_root=dataset_root)
    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    # lstm = LSTM(train_data={"X": train_X, "y": train_y},
    #             test_data={"X": test_X, "y": test_y})

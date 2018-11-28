'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 9:38:33 pm
Author: ankurrc
'''
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard

from model import Model
from data import Dataset
from metrics import F1_metrics


class LSTM_model(Model):

    def __init__(self, train_data=None, test_data=None):
        super().__init__(train_data=train_data, test_data=test_data)
        self.build()

    def evaluate(self, log_dir=None):
        callbacks = [F1_metrics(), TensorBoard(
            log_dir=log_dir, write_grads=True, write_graph=True, histogram_freq=3)]
        self.model.fit(self.train_X, self.train_y, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=self.verbose, validation_split=0.2, callbacks=callbacks)
        # evaluate model
        _, accuracy = self.model.evaluate(
            self.test_X, self.test_y, batch_size=self.batch_size, verbose=self.verbose)
        return accuracy

    def build(self):
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(
            self.n_timesteps, self.n_features)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.n_outputs, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])


if __name__ == "__main__":
    dataset_root = "/media/ankurrc/new_volume/633_ml/project/code/dataset/UCI HAR Dataset/"
    dataset = Dataset(dataset_root=dataset_root)
    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    lstm = LSTM(train_data={"X": train_X, "y": train_y},
                test_data={"X": test_X, "y": test_y})

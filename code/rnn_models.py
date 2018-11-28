'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 9:38:33 pm
Author: ankurrc
'''
from keras.layers import Sequential, LSTM, Dense, Dropout

from model import Model
from data import Dataset


class LSTM(Model):

    def __init__(self, train_data=None, test_data=None):
        super().__init__(train_data=train_data, test_data=test_data)

    def evaluate(self):

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

import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import rmsprop
from keras.utils import to_categorical

# have UCI HAR dataset in same directory as this file
path = os.getcwd()
dataset_path = '/UCI HAR Dataset'
train_path = '/train'
test_path = '/test'
x_path = '/Inertial Signals/'
y_path_train = '/y_train.txt'
y_path_test = '/y_test.txt'
# get raw training data from files
x_train_data_path = path + dataset_path + train_path + x_path
x_train = []
for file in sorted(os.listdir(x_train_data_path)):
  x_train.append(pd.read_csv(x_train_data_path+file, header=None, delim_whitespace=True))
y_train_data_path = path + dataset_path + train_path + y_path_train
y_train = pd.read_csv(y_train_data_path, header=None, delim_whitespace=True)
# get raw testing data from files
x_test_data_path = path + dataset_path + test_path + x_path
x_test = []
for file in sorted(os.listdir(x_test_data_path)):
  x_test.append(pd.read_csv(x_test_data_path+file, header=None, delim_whitespace=True))
y_test_data_path = path + dataset_path + test_path + y_path_test
y_test = pd.read_csv(y_test_data_path, header=None, delim_whitespace=True)
# wrap data to use in model
x_test = np.dstack(x_test)
x_train = np.dstack(x_train)
y_test = to_categorical(y_test-1)
y_train = to_categorical(y_train-1)
# get data dimensions
samples = x_train.shape[1]
features = x_train.shape[2]
outputs = y_train.shape[1]
# create cnn model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', dilation_rate=2, input_shape=(samples,features)))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', dilation_rate=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(outputs, activation='softmax'))
opt = rmsprop(lr=0.001, rho =0.95, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#
verbose = 1
epochs = 20
batch_size = 32
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
print('accuracy', accuracy)
print('loss', loss)
model.save(path+'/model.h5')
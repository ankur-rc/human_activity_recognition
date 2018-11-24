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
def run_cnn(x_train,y_train,x_test,y_test,prev_accuracy):
	# get data dimensions
	samples = x_train.shape[1]
	features = x_train.shape[2]
	outputs = y_train.shape[1]
	# create cnn model
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=8, activation='relu', dilation_rate=1, input_shape=(samples,features)))
	model.add(Dropout(0.5))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', dilation_rate=2))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', dilation_rate=4))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', dilation_rate=8))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(outputs, activation='softmax'))
	opt = rmsprop(lr=0.001, rho =0.99, epsilon=None, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#
	verbose = 1
	epochs = 20
	batch_size = 32
	model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
	loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
	print('accuracy', accuracy)
	print('loss', loss)
	if accuracy > prev_accuracy:
		model.save(path+'/model.h5')
	return accuracy
prev_accuracy = 0
accuracy = []
for i in range(10):
	prev_accuracy = run_cnn(x_train,y_train,x_test,y_test,prev_accuracy)
	accuracy.append(prev_accuracy)
print('accuracy',accuracy)
print('mean accuracy',np.mean(accuracy))
print('stdev accuracy',np.std(accuracy))

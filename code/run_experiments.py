'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 10:55:41 pm
Author: ankurrc
'''
import numpy as np
import os

from rnn_models import LSTM_model, CNN_LSTM_model, ConvLSTM_model
from cnn_models import Simple_shallow_cnn, Wavenet_deep_cnn
from data import Dataset

from keras import backend as K


def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: {:.3f}% (+/-{:.3f})'.format(m, s))


def run_experiment(repeats=10, model_type=None, train_data=None, test_data=None, tb_log_dir=None):

    scores = list()
    for r in range(repeats):
        K.clear_session()
        _log_dir = os.path.join(tb_log_dir, "run_{}".format(r))
        model = get_model(name=model_type, log_dir=_log_dir,
                          train_data=train_data, test_data=test_data)
        score = model.evaluate(log_dir=_log_dir)
        score = score * 100.0
        print('>>>>> # {:d}: {:.3f}'.format(r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


def get_model(name, log_dir=None, train_data=None, test_data=None):
    if name is 'lstm':
        model = LSTM_model(train_data=train_data,
                           test_data=test_data, tb_log_dir=log_dir)
    elif name is 'cnn_lstm':
        model = CNN_LSTM_model(train_data=train_data,
                               test_data=test_data, tb_log_dir=log_dir)
    elif name is 'conv_lstm':
        model = ConvLSTM_model(train_data=train_data,
                               test_data=test_data, tb_log_dir=log_dir)
    elif name is 'simple_cnn':
        model = Simple_shallow_cnn(train_data=train_data,
                               test_data=test_data, tb_log_dir=log_dir)
    elif name is 'wavenet_cnn':
        model = Wavenet_deep_cnn(train_data=train_data,
                               test_data=test_data, tb_log_dir=log_dir)
    else:
        raise KeyError("Key '{}' not implemented!".format(name))

    return model


def main():
    path = os.getcwd()
    dataset_path = '/UCI HAR Dataset'
    dataset_root = path + dataset_path
    log_dir = "logs"
    num_repeats = 10

    dataset = Dataset(dataset_root=dataset_root)
    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    models = ['wavenet_cnn','simple_cnn']

    for model_type in models:
        _log_dir = os.path.join(log_dir, model_type)
        run_experiment(repeats=num_repeats, model_type=model_type, train_data={"X": train_X, "y": train_y},
                       test_data={"X": test_X, "y": test_y}, tb_log_dir=_log_dir)


if __name__ == "__main__":
    main()

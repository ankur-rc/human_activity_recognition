'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 10:55:41 pm
Author: ankurrc
'''
import numpy as
import os

from rnn_models import LSTM_model
from data import Dataset


def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: {:.3f%} (+/-{:.3f})'.format(m, s))


def run_experiment(repeats=10, model=None, log_dir=None):

    scores = list()
    for r in range(repeats):
        _log_dir = os.path.join(log_dir, "run_{}".format(r))
        score = model.evaluate(log_dir=_log_dir)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


def main():
    dataset_root = "/media/ankurrc/new_volume/633_ml/project/code/dataset/UCI HAR Dataset/"
    log_dir = "media/ankurrc/new_volume/633_ml/project/code/log/"

    dataset = Dataset(dataset_root=dataset_root)
    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    models = ['lstm']

    for model in models:
        if model is 'lstm':
            model = LSTM_model(train_data={"X": train_X, "y": train_y},
                               test_data={"X": test_X, "y": test_y})
        else:
            raise KeyError("Key '{}' not implemented!".format(model))

        _log_dir = os.path.join(log_dir, model)

        run_experiment(model=model, log_dir=_log_dir)


if __name__ == "__main__":
    main()

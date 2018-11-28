'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 10:55:41 pm
Author: ankurrc
'''
import numpy as np

from rnn_models import LSTM_model
from data import Dataset


def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def run_experiment(repeats=10, train_data=None, test_data=None):

    scores = list()
    for r in range(repeats):
        model = LSTM_model(train_data=train_data, test_data=test_data)
        score = model.evaluate()
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


def main():
    dataset_root = "/media/ankurrc/new_volume/633_ml/project/code/dataset/UCI HAR Dataset/"
    dataset = Dataset(dataset_root=dataset_root)
    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    run_experiment(train_data={"X": train_X, "y": train_y},
                   test_data={"X": test_X, "y": test_y})


if __name__ == "__main__":
    main()

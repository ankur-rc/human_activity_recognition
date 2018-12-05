'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 10:55:41 pm
Author: ankurrc
'''
import numpy as np
import os
import argparse

from rnn_models import LSTM_model, CNN_LSTM_model, ConvLSTM_model
from cnn_models import Simple_shallow_cnn, Wavenet_deep_cnn
from data import Dataset

from keras import backend as K


def summarize_results(precison, recall, f1):
    f1_m, f1_s = np.mean(f1), np.std(f1)
    pre_m, pre_s = np.mean(precison), np.std(precison)
    re_m, re_s = np.mean(recall), np.std(recall)

    print(
        'Precision: {:.5f} (+/-{:.5f}) \t Recall: {:.5f} (+/-{:.5f}) \t F1: {:.5f} (+/-{:.5f})'.format(pre_m, pre_s, re_m, re_s, f1_m, f1_s))

    return (pre_m, pre_s), (re_m, re_s), (f1_m, f1_s)


def run_experiment(repeats=10, model_type=None, train_data=None, test_data=None, tb_log_dir=None):

    f1s = []
    precisions = []
    recalls = []

    model = None
    for r in range(repeats):
        K.clear_session()
        _log_dir = os.path.join(tb_log_dir, "run_{}".format(r))
        model = get_model(name=model_type, log_dir=_log_dir,
                          train_data=train_data, test_data=test_data)
        precision, recall, f1 = model.evaluate(log_dir=_log_dir)
        print('>>>>> #{}--> Precision: {:.5f}, Recall: {:.5f}, F1: {:.5f}'.format(r +
                                                                                  1, precision, recall, f1))
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    # summarize results
    p, r, f1 = summarize_results(precisions, recalls, f1s)
    return p, r, f1, model


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


def main(args):
    dataset_root = args.dataset
    num_repeats = args.repeats
    models = args.models

    log_dir = "logs"
    results_dir = "results"
    models_dir = "models"

    dataset = Dataset(dataset_root=dataset_root)
    train_X, train_y = dataset.load()
    test_X, test_y = dataset.load(split="test")

    for model_type in models:
        print(">>>>>>>>>>>>> Running experiments for '{}'".format(model_type))
        _log_dir = os.path.join(log_dir, model_type)
        precision, recall, f1, model = run_experiment(repeats=num_repeats, model_type=model_type, train_data={"X": train_X, "y": train_y},
                                                      test_data={"X": test_X, "y": test_y}, tb_log_dir=_log_dir)
        print(">>>>>>>>>>>>> Writing results for '{}'".format(model_type))
        with open(os.path.join(results_dir, model_type + ".txt"), "w") as res:
            line = "{}:\n".format(model_type)
            line += "Precision: {:.5f} (+/-{:.5f}) \n Recall: {:.5f} (+/-{:.5f}) \n F1: {:.5f} (+/-{:.5f})\n".format(precision[0], precision[1],
                                                                                                                     recall[0], recall[1], f1[0], f1[1])
            line += "--------------------------------------------------------------------------------------------------------------------------------- \n"
            res.writelines(line)
        print(">>>>>>>>>>>>> Saving the model: {}.h5".format(model_type))
        model.model.save(os.path.join(models_dir, model_type + ".h5"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run models on the UCI HAR dataset.")
    parser.add_argument(
        "--dataset", help="Root path to UCI HAR dataset", type=str, default="UCI HAR Dataset/")
    parser.add_argument(
        "--repeats", help="No. of repeats for each model", type=int, default=10)
    parser.add_argument("--models", help="List of models to evaluate on. Valid models are: [lstm, cnn_lstm, conv_lstm, simple_cnn, wavenet_cnn]",
                        nargs='+', type=str)

    args = parser.parse_args()
    main(args)

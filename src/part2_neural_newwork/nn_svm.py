from libsvm.svmutil import *

import pandas as pd
import numpy as np
import os

# For plotting
import matplotlib
import matplotlib.pyplot as plt


def data_loader(data_dir, mode):
    def normalize(x):
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / x_std
        return x

    data = pd.read_csv(data_dir, header=0, engine="c").values
    feats = list(range(3))
    if mode == 'test':
        data = data[:, feats]
        data = normalize(data)
        return data
    else:
        target = (data[:, -1])
        data = data[:, feats]
        data = normalize(data)
        return data, target

def plot_pred(preds, targets):
    plt.figure()
    plt.scatter(targets, preds, c='r', alpha=0.5, label="Predicted data")
    plt.plot([-900, 180], [-900, 180], c='b', label="Theoretical data")
    plt.xlim(-900, 180)
    plt.ylim(-900, 180)
    plt.xlabel('Phase by CST')
    plt.ylabel('Phase by M2LP')
    plt.title('Prediction Error Curve')
    plt.legend(loc='best')
    plt.savefig(r"../../img/nn_model/svr_prediction_error.png")
    plt.show()

if __name__ == "__main__":
    config = {
        # model
        'hype_para': '-s 3 -t 2 -c 1 -g 0.1',
        # path
        'model_path': r'./models/model.pth',
        'tr_path': r'../../data/dataset/tr_set_unwrap.csv',
        've_path': r'../../data/dataset/ve_set_unwrap.csv',
        'tt_path': r'../../data/dataset/tt_set.csv'
    }

    data_tr, target_tr = data_loader(config['tr_path'], 'train')
    data_ve, target_ve = data_loader(config['ve_path'], 'verify')
    data_tt = data_loader(config['tt_path'], 'test')

    model = svm_train(target_tr, data_tr, config['hype_para'])

    p_label, p_acc, p_val = svm_predict([0, 0], data_ve, model)

    plot_pred(p_label, target_ve)

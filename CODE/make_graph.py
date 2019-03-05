#
# CO(GT)
# PT08.S1(CO)
# NMHC(GT)
# C6H6(GT)
# PT08.S2(NMHC)
# NOx(GT)
# PT08.S3(NOx)
# NO2(GT)
# PT08.S4(NO2)
# PT08.S5(O3)
# T
# RH
# AH

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "None"

import random as rd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy import stats, io
from pylab import savefig

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, recall_score,
                            roc_curve, recall_score, classification_report, f1_score,
                            precision_recall_fscore_support, accuracy_score, precision_score)

from keras import regularizers, optimizers
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model, load_model, model_from_json



## 그래프 그리기
def show_graph():

    fig = plt.figure(figsize=(14, 9), facecolor='lightgray')

    # original_axes = plt.subplot2grid((4,4), (0,0), colspan=4)
    outlier_axes = plt.subplot2grid((3,4), (0,0), colspan=4)
    incremental_axes = plt.subplot2grid((3,4), (1,0), colspan=2)
    sudden_axes = plt.subplot2grid((3,4), (1,2), colspan=2)
    recurring_1_axes = plt.subplot2grid((3,4), (2,0), colspan=2)
    recurring_2_axes = plt.subplot2grid((3,4), (2,2), colspan=2)

    axes = [outlier_axes,
            # original_axes,
            incremental_axes,
            sudden_axes,
            recurring_1_axes,
            recurring_2_axes]

    plt.suptitle('{} of Data'.format(cname), fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9,
                        bottom=0.05,
                        left=0.08,
                        right=0.95,
                        hspace=0.5,
                        wspace=0.5)

    nums = [i for i in range(len(types))]
    for ax, title, num in zip(axes, types, nums):

        data = pd.read_csv('dataset/outlier/{}.csv'.format(title))
        data = data[cname]

        ax.set_title('%s case'%(title), fontsize=15)
        # ax.set_ylabel('Mean Squared Error', fontsize=8)
        ax.set_ylim(-0.001, 5000)

        ax.plot(data, marker='o', ms=2, linestyle='')

    savefig('1016/outlier_data.png')
    plt.show()


def main():
    show_graph()


if __name__ == '__main__':

    types = ['outlier',
            # 'original',
            'incremental',
            'sudden',
            'recurring_1',
            'recurring_2']

    cname = 'PT08.S1(CO)'

    main()












print('FINE')

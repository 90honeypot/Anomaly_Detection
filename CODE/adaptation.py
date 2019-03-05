## import
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
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard

# ---------------------------------------------------------------------------- #
## original 데이터의 MSE를 계산하여 가장 큰 값을 threshold로 설정
def calc_threshold(model, data):
    ret = .0
    length = len(data)

    X = normalize(data)
    pred = model.predict(X)
    thres = np.mean(np.power(X - pred, 2), axis=1)
    for e in thres:
        ret = e if ret < e else ret
    print('mean threhold({})'.format(np.mean(thres)))
    print('finish calculating threshold({})'.format(ret))
    return ret

## 저장된 모델을 불러옴
def load_model(name):

    json_file = open('%s.json'%(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights('%s.h5'%(name))

    sgd = optimizers.SGD(lr=learning_rate, momentum=0.0)
    model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])

    return model

## model에 sliding_window의 데이터를 fit함
def fitting_data(model, data):
    print('fit model')
    X_train = normalize(data)
    history = model.fit(X_train, X_train,
                        epochs = epoch,
                        batch_size = batch_size,
                        shuffle = False,
                        verbose = 0).history
    return model

## threhold와 data를 받아서 outlier를 detect하고 fitting함
def handling_data(model, data, thres):
    '''
        * thres : threshold values
    '''
    length = len(data) # 데이터 길이
    # 기본 슬라이딩 윈도우 셋팅
    sliding_window = data.take(np.random.permutation(length)[:sliding_window_size])

    MSEs = []
    pred_class = []
    thresholds = []
    for i in range(length):
        '''
            * 데이터 1개씩 들어올때마다 detect 함
            * sliding_window에 추가해야하는 데이터의 개수가 되면 sliding_window 업데이트 후 fitting
        '''
        # detect
        X = normalize(data.loc[i:i])
        pred = model.predict(X)
        MSE_X = np.mean(np.power(X - pred, 2), axis=1)
        MSEs.append(MSE_X)
        # threshold값보다 큰 data는 outlier로 predict함 (class = 1)
        pred = 1 if MSE_X > thres else 0
        pred_class.append(pred)
        thresholds.append(thres)
        # 슬라이딩 윈도우 다 차면 fitting
        if (i != 0 and i % adding_size == adding_size-1) or i == length-1:
            remain_window = sliding_window.take(np.random.permutation(sliding_window_size)[:sliding_window_size - adding_size])
            sliding_window = shuffle(pd.concat([remain_window, data.loc[i-adding_size+1:i]]))
            print('{} ~ {}'.format(i-adding_size+1, i))
            # temp = np.mean(MSEs[i-adding_size+1:i])
            # threshold = temp
            # thresholds[i] = threshold*0.99 + temp*0.01
            model = fitting_data(model, sliding_window)

    # 계산된 MSE값을 리턴함
    # print(thresholds)
    return MSEs, pred_class, thresholds

## 그래프 그리기
def show_graph(types, MSE_groups, length, thresholds):

    fig = plt.figure(figsize=(14, 9), facecolor='lightgray')

    original_axes = plt.subplot2grid((4,4), (0,0), colspan=4)
    outlier_axes = plt.subplot2grid((4,4), (1,0), colspan=4)
    incremental_axes = plt.subplot2grid((4,4), (2,0), colspan=2)
    sudden_axes = plt.subplot2grid((4,4), (2,2), colspan=2)
    recurring_1_axes = plt.subplot2grid((4,4), (3,0), colspan=2)
    recurring_2_axes = plt.subplot2grid((4,4), (3,2), colspan=2)

    axes = [original_axes,
            outlier_axes,
            incremental_axes,
            sudden_axes,
            recurring_1_axes,
            recurring_2_axes]

    plt.suptitle('Comparison of MSE for each case', fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9,
                        bottom=0.05,
                        left=0.08,
                        right=0.95,
                        hspace=0.5,
                        wspace=0.5)

    nums = [i for i in range(len(axes))]
    for ax, title, num in zip(axes, types, nums):
        f1score = f1_score(MSE_groups[num].true_class.values, MSE_groups[num].pred_class.values, average='macro')
        ax.set_title('%s case'%(title), fontsize=15)
        ax.set_ylabel('Reconstruction Error', fontsize=8)
        ax.set_ylim(-0.001, 0.025)
        mean_MSE = []
        flag = 0
        for i in range(length):
            if i % 100 == 99 or i == length-1:
                mean_m = np.mean(MSE_groups[num]['mse'].loc[flag:i])
                flag = i+1
                mean_MSE.append(mean_m)

        ###############################################################################################################
        # ax.hlines(threshold, 0, length, colors="r", zorder=100, label='Threshold')
        groups = MSE_groups[num].groupby('true_class')
        for name, group in groups:
            ax.plot(group.index, group.mse, marker='o', ms=2, linestyle="", label='Outlier' if name == 1 else 'Normal')
        ax.plot(thresholds, 'r', label='threshold')
        ###############################################################################################################

        # ax.plot(mean_MSE)

    savefig('1016/2_shit.png')
    plt.show()



def main():
    '''
        * MSE_df : data의 정보를 저장할 DataFrame
                   index / MSE / true_class / pred_class / threshold
    '''

    MSE_original = pd.DataFrame(columns=['mse', 'true_class', 'pred_class', 'threshold'])
    MSE_outlier = pd.DataFrame(columns=['mse', 'true_class', 'pred_class', 'threshold'])
    MSE_incremental = pd.DataFrame(columns=['mse', 'true_class', 'pred_class', 'threshold'])
    MSE_sudden = pd.DataFrame(columns=['mse', 'true_class', 'pred_class', 'threshold'])
    MSE_recurring_1 = pd.DataFrame(columns=['mse', 'true_class', 'pred_class', 'threshold'])
    MSE_recurring_2 = pd.DataFrame(columns=['mse', 'true_class', 'pred_class', 'threshold'])

    MSE_groups = [MSE_original,
                  MSE_outlier,
                  MSE_incremental,
                  MSE_sudden,
                  MSE_recurring_1,
                  MSE_recurring_2]

    nums = [i for i in range(len(MSE_groups))]

    # normal data의 threshold 값 계산
    # normal data 중 가장 큰 reconstruction error 값을 가지는 놈
    original_data = pd.read_csv('dataset/outlier/original.csv').drop(['Class'], axis=1)
    length = len(original_data)
    autoencoder = load_model('autoencoder(2000)')
    threshold = calc_threshold(autoencoder, original_data) + 0.00001

    for name, i in zip(types, nums):
        print('### [data : {}] start ###'.format(name))
        # 데이터 읽기
        data = pd.read_csv('dataset/outlier/{}.csv'.format(name))

        ####### test용 #######
        # data = data.loc[0:999]
        # length = len(data)
        ######################

        # 데이터의 class(label) 저장
        MSE_groups[i]["true_class"] = data['Class']
        # 데이터의 class 버리고
        data = data.drop(['Class'], axis=1)
        autoencoder = load_model('test_model_1')

        mse, pred_class, thresholds = handling_data(autoencoder, data, threshold)
        MSE_groups[i]['mse'] = mse
        MSE_groups[i]['pred_class'] = pred_class

    show_graph(types, MSE_groups, length, thresholds)

    print('finish')


if __name__ == '__main__':

    ## Global Variables
    epoch = 1
    batch_size = 32
    loss = 'mean_squared_error'
    new_ratio = 0.6
    # new_ratios = [i/10 for i in range(1, 10)]

    learning_rate = 0.1
    # learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    sliding_window_size = 100
    # sliding_window_sizes = [100, 1000, 10000]

    adding_size = round(sliding_window_size * new_ratio)

    types = ['original',
            'outlier',
            'incremental',
            'sudden',
            'recurring_1',
            'recurring_2']


    main()




# FINE
print('# --- FINISH --- #')

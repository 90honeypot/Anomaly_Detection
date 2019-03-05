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
def load_model(name, learning_rate):

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
def handling_data(model, data, thres, s_size, nr):
    '''
        * thres : threshold values
    '''
    adding_size = round(s_size * nr)
    length = len(data) # 데이터 길이
    # 기본 슬라이딩 윈도우 셋팅
    sliding_window = data.take(np.random.permutation(length)[:s_size])

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
        if (i != 0 and i % s_size == s_size-1) or i == length-1:
            remain_window = sliding_window.take(np.random.permutation(s_size)[:s_size - adding_size])
            adding_window = data.loc[i-s_size+1:i].take(np.random.permutation(s_size)[:adding_size])
            sliding_window = shuffle(pd.concat([remain_window, adding_window]))
            print('{} ~ {}'.format(i-s_size+1, i))
            # temp = np.mean(MSEs[i-adding_size+1:i])
            # threshold = temp
            # thresholds[i] = threshold*0.99 + temp*0.01
            model = fitting_data(model, sliding_window)

    # 계산된 MSE값을 리턴함
    # print(thresholds)
    return MSEs, pred_class, thresholds

## 그래프 그리기
def show_graph(MSE_groups, length):
    cases = ['S-Model', 'R-Model']
    for m, title in zip(MSE_groups, cases):
        plt.figure(figsize=(10, 5), facecolor='lightgray')

        f1score = f1_score(m.true_class.values, m.pred_class.values, average='macro')

        plt.title('outlier detection of %s about I-data (F1 score %.3f)'%(title, f1score), fontsize=18)
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.8,
        #                     bottom=0.05,
        #                     left=0.08,
        #                     right=0.95,
        #                     hspace=0.5,
        #                     wspace=0.2)
        # nums = [i for i in range(len(axes))]
        # for ax, num, title in zip(axes, nums, titles):
        plt.ylabel('Reconstruction Error', fontsize=8)
        plt.ylim(-0.001, 0.025)
        mean_MSE = []
        flag = 0
        for i in range(length):
            if i % 100 == 99 or i == length-1:
                mean_m = np.mean(m['mse'].loc[flag:i])
                flag = i+1
                mean_MSE.append(mean_m)

        ###############################################################################################################
        # ax.hlines(threshold, 0, length, colors="r", zorder=100, label='Threshold')
        groups = m.groupby('true_class')
        for name, group in groups:
            plt.plot(group.index, group.mse, marker='o', ms=2, linestyle="", label='Outlier' if name == 1 else 'Normal')
        plt.plot(m.threshold, 'r', label='threshold')
        ###############################################################################################################

        # ax.plot(mean_MSE)

        savefig('test(%s-%s).png'%(type, title))
    plt.show()



def main():
    '''
        * MSE_df : data의 정보를 저장할 DataFrame
                   index / MSE / true_class / pred_class / threshold
    '''

    MSE_case_1 = pd.DataFrame(columns=['mse', 'true_class', 'pred_class', 'threshold'])
    MSE_case_2 = pd.DataFrame(columns=['mse', 'true_class', 'pred_class', 'threshold'])
    MSE_groups = [MSE_case_1,
                  MSE_case_2]
    nums = [i for i in range(len(learning_rate))]

    # normal data의 threshold 값 계산
    # normal data 중 가장 큰 reconstruction error 값을 가지는 놈
    original_data = pd.read_csv('dataset/outlier/original.csv').drop(['Class'], axis=1)
    length = len(original_data)
    autoencoder = load_model('autoencoder(2000)', 0.01)
    threshold = calc_threshold(autoencoder, original_data) + 0.0001

    for lr, sws, i, nr in zip(learning_rate, sliding_window_size, nums, new_ratios):
        print('### [data : {}] start ###'.format(type))
        # 데이터 읽기
        data = pd.read_csv('dataset/outlier/{}.csv'.format(type))

        # 데이터의 class(label) 저장
        MSE_groups[i]["true_class"] = data['Class']
        # 데이터의 class 버리고
        data = data.drop(['Class'], axis=1)
        autoencoder = load_model('test_model_1', lr)

        mse, pred_class, thresholds = handling_data(autoencoder, data, threshold, sws, nr)
        MSE_groups[i]['mse'] = mse
        MSE_groups[i]['pred_class'] = pred_class
        MSE_groups[i]['threshold'] = thresholds

    show_graph(MSE_groups, length)

    print('finish')


if __name__ == '__main__':

    ## Global Variables
    epoch = 1
    batch_size = 32
    loss = 'mean_squared_error'
    new_ratios = [0.8, 0.2]
    # new_ratios = [i/10 for i in range(1, 10)]

    learning_rate = [0.001, 0.1]
    # learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    sliding_window_size = [10000, 100]
    # sliding_window_sizes = [100, 1000, 10000]
                         

    type = 'recurring_1'


    main()




# FINE
print('# --- FINISH --- #')

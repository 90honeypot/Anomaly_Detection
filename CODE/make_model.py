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
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard

data_name = 'original.csv'
epoch = 1
batch_size = 32
loss = 'mean_squared_error'
threshold_alpha = 0.125
new_ratio = 0.6
learning_rates = [0.01]
data_split = 10


threshold = 0.0
## data load
data = pd.read_csv("dataset/outlier/%s" % (data_name))
data = data.drop(['Class'], axis=1)
# data = data.drop(['Date', 'Time'], axis=1)
header_list = data.columns.tolist()
input_dim = data.shape[1]
data_length = len(data)

# sliding_window_size = round(len(data) / data_split)
sliding_window_size = 1000
## the number of test
for test in range(1):

    reconstruction_error = []

    ## experiment case
    for learning_rate in learning_rates:

        ## make model and complie
        input_layer = Input(shape=(input_dim, ))
        encoder = Dense(input_dim*3//4, activation="tanh",
                        activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(input_dim*9//16, activation="relu")(encoder)
        encoder = Dense(input_dim*9//32, activation="relu")(encoder)
        decoder = Dense(input_dim*9//16, activation="relu")(encoder)
        decoder = Dense(input_dim*3//4, activation="relu")(encoder)
        decoder = Dense(input_dim, activation="tanh")(encoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.0)
        autoencoder.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])

        adding_index = round(sliding_window_size * new_ratio)
        index = 0

        sum_mse = []
        re = []
        flag = 0
        count = 0
        ## initialization of sliding window
        sliding_window = shuffle(data.take(np.random.permutation(data_length))[:sliding_window_size])

        for time_step in range(10000000):
            ## input data in sliding window
            if index + adding_index <= data_length:
                adding_window = data.loc[index : index + adding_index - 1]
                index += adding_index
            else:
                adding_window = data.loc[index : data_length - 1]
                index = sliding_window_size - (data_length - index)
                temp = data.loc[0 : index - 1]
                adding_window = pd.concat([adding_window, temp])
                count += 1
                print('count = ', count)
            remain_window = sliding_window.take(np.random.permutation(sliding_window_size)[:sliding_window_size - adding_index])
            sliding_window = shuffle(pd.concat([remain_window, adding_window]))

            ## X_train is the data for fitting
            X_train = normalize(sliding_window)
            ## fitting
            history = autoencoder.fit(X_train, X_train,
                                        epochs = epoch,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        verbose = 0).history


            ## graph when finished fitting the entire data 50 times
            if count == 50:
                ## save model
                autoencoder_json = autoencoder.to_json()
                with open("test_model_1.json", "w") as json_file:
                    json_file.write(autoencoder_json)
                autoencoder.save_weights("test_model_1.h5")

                break;


print('## MODEL SAVE ##')

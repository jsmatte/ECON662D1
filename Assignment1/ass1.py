#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Assignment 1, Q1.2
"""

import time
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt

def plot(tickLoc, tickLables, residuals, title, filename, save):
    plt.figure(figsize=(18, 8))
    plt.hlines(0, tickLoc[0], tickLoc[-1])
    plt.scatter(tickLoc, residuals, c = 'r', label = 'residuals')
    plt.xlabel('time')
    plt.xticks(ticks = tickLoc, labels = tickLables, rotation = 90)
    plt.ylabel('Income')
    plt.legend(loc = 'upper right')
    plt.title(str(title))
    if save:
        plt.savefig(str(out_dir + filename))
    plt.show()
    plt.close()

    return None

def regression(y, C, ticks):
    # Regressing C on Y as a function of time
    reg_model = LinearRegression()
    y_array = np.array([[j] for j in y])
    reg_model.fit(y_array, C)

    r_2 = r2_score(C, reg_model.predict(y_array))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(y, C)

    predictions = reg_model.predict(y_array)
    residuals = []
    for i in range(len(y)):
        temp_resid = C[i] - predictions[i]
        residuals.append(temp_resid)

    tick_loc = [x for x in range(len(ticks))]

    title = 'Linear regression residuals as a function of time'
    flnm = '/regression.pdf'
    save = False
    plot(tick_loc, ticks, residuals, flnm, title, save)

    return None


def log_regression(y, C, ticks):
    # log-linear regression of logC on logY
    reg_model = LinearRegression()
    y_array = np.array([[np.log(j)] for j in y])
    log_C = np.array([np.log(c) for c in C])
    reg_model.fit(y_array, log_C)

    r_2 = r2_score(log_C, reg_model.predict(y_array))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(y, C)

    predictions = reg_model.predict(y_array)
    residuals = []
    for i in range(len(y)):
        temp_resid = log_C[i] - predictions[i]
        residuals.append(temp_resid)

    tick_loc = [x for x in range(len(ticks))]

    title = 'Log-linear regression residuals as a function of time'
    flnm = '/log_regression.pdf'
    save = False
    plot(tick_loc, ticks, residuals, title, flnm, save)

    return None


if __name__ == "__main__":
    start = time.time()

    in_dir = '/Users/jsmatte/github/ECON662D1/Assignment1/data'

    # household final consumption file
    hhfc_file = str(in_dir + '/3610010701-eng-prepro.csv')

    # household income file
    hhi_file = str(in_dir + '/3610043501-eng-prepro.csv')

    # load files as pd
    hhfc_df = pd.read_csv(hhfc_file, sep = ',', header = 0)
    # print(hhfc_df)
    # print(hhfc_df.Estimates.values)
    # print(hhfc_df.dtypes)

    hhi_df = pd.read_csv(hhi_file, sep = ',', header = 0)
    # print(hhi_df.Estimates.values)

    # get the relevant data from each df
    # Consumption expenditures class -> C -> df column
    # Income -> Y
    y = hhi_df.T.iloc[:, 0].values[1:]
    C = hhfc_df.T.iloc[:, 55].values[1:]

    tick_labels = hhi_df.columns.values[1:]

    regression(y, C, tick_labels)
    log_regression(y, C, tick_labels)


    # # test
    # diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    # diabetes_X = diabetes_X[:, np.newaxis, 2]
    # diabetes_X_train = diabetes_X[:-20]
    # diabetes_y_train = diabetes_y[:-20]
    # print(diabetes_X_train.shape)
    # print(diabetes_y_train.shape)


    end = time.time()
    print(end - start)

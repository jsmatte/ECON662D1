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


def plot_reg(y, C, y_array, predictions, filename, save):
    plt.plot(y_array, predictions, label = 'regression line')
    plt.scatter(y, C, c = 'r')
    plt.xlabel('income')
    plt.ylabel('consumption')
    plt.legend(loc = 'upper left')
    if save:
        plt.savefig(str(out_dir + filename))
    plt.show()
    plt.close()

    return None


def plot(tickLoc, tickLables, residuals, title, filename, save):
    plt.figure(figsize=(24, 8))
    plt.hlines(0, tickLoc[0], tickLoc[-1])
    plt.scatter(tickLoc, residuals, c = 'r', label = 'residuals')
    plt.xlabel('Time')
    plt.xticks(ticks = tickLoc, labels = tickLables, rotation = 90)
    plt.ylabel('Consumption (x$1,000,000, 2012 dollars)')
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
    C_array = np.array([[i] for i in C])
    reg_model.fit(y_array, C)
    # print(reg_model.coef_, reg_model.intercept_)

    r_2 = r2_score(C, reg_model.predict(y_array))

    predictions = reg_model.predict(y_array)

    residuals = []
    for i in range(len(y)):
        temp_resid = C[i] - predictions[i]
        residuals.append(temp_resid)

    tick_loc = [x for x in range(len(ticks))]

    title = 'Residuals of Linear Regression as a Function of Time'
    flnm = '/regression.pdf'
    save = True
    plot(tick_loc, ticks, residuals, title, flnm, save)

    return None


def log_regression(y, C, ticks):
    # log-linear regression of logC on logY
    log_reg_model = LinearRegression()
    log_y_array = np.array([[np.log(j)] for j in y])
    log_C = np.array([np.log(c) for c in C])
    log_reg_model.fit(log_y_array, log_C)
    # print(log_reg_model.coef_, log_reg_model.intercept_)

    r_2 = r2_score(log_C, log_reg_model.predict(log_y_array))

    log_predictions = log_reg_model.predict(log_y_array)

    log_residuals = []
    for i in range(len(y)):
        temp_resid = log_C[i] - log_predictions[i]
        log_residuals.append(temp_resid)

    tick_loc = [x for x in range(len(ticks))]

    title = 'Residuals of Log-linear Regression as a Function of Time'
    flnm = '/log_regression.pdf'
    save = True
    plot(tick_loc, ticks, log_residuals, title, flnm, save)

    return None


if __name__ == "__main__":
    start = time.time()

    in_dir = '/Users/jsmatte/github/ECON662D1/Assignment1/data'
    out_dir = '/Users/jsmatte/Documents/Tex/ECON662D1/Assignment1'

    # household final consumption file
    hhfc_file = str(in_dir + '/3610010701-eng-prepro.csv')

    # household income file
    hhi_file = str(in_dir + '/3610043501-eng-prepro.csv')

    # load files as pd
    hhfc_df = pd.read_csv(hhfc_file, sep = ',', header = 0)
    hhi_df = pd.read_csv(hhi_file, sep = ',', header = 0)

    # get the relevant data from each df
    # Consumption expenditures class -> C -> df column
    # Income -> Y
    y = hhi_df.T.iloc[:, 0].values[1:]
    C = hhfc_df.T.iloc[:, 55].values[1:]

    tick_labels = hhi_df.columns.values[1:]

    regression(y, C, tick_labels)
    log_regression(y, C, tick_labels)


    end = time.time()
    print(end - start)

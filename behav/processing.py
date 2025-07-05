import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
import warnings
warnings.simplefilter("ignore")

def make_psych_table(processed_data, optoStim = 'none'):
    """
    For a given optoStim, makes a psychometric table of the following form:
    | Intensity     | Lick L       | Lick R       |
    | ------------- | ------------ | ------------ |
    | <intensity 1> | <fraction L> | <fraction R> |
    | <intensity 2> | <fraction L> | <fraction R> |
    If no optoStim provided, considers data for no optogenetic stimulation.
    """
    opto_data = processed_data[processed_data['optoStim'] == optoStim]

    lick_L = (opto_data['trial_type'] == 3) | (opto_data['trial_type'] == 20)
    lick_R = (opto_data['trial_type'] == 5) | (opto_data['trial_type'] == 18)

    no_of_intensities = len(set(opto_data['stim_left']))

    psych_table = np.zeros((no_of_intensities, 3))

    for index in range(no_of_intensities):
        psych_table[index, 0] = list(set(opto_data['stim_left']))[index]
        psych_table[index, 1] = len(opto_data[lick_L]
                                    [opto_data['stim_left'] == psych_table[index, 0]])
        psych_table[index, 1] = psych_table[index, 1] / len(opto_data[opto_data['stim_left'] == psych_table[index, 0]])
        psych_table[index, 2] = len(opto_data[lick_R]
                                    [opto_data['stim_left'] == psych_table[index, 0]])
        psych_table[index, 2] = psych_table[index, 2] / len(opto_data[opto_data['stim_left'] == psych_table[index, 0]])
        
    psych_table = np.array(psych_table)

    return psych_table


def take_needed_ints_only(psych_table, max_val = 1.2):
    """
    Given psych_table, only keeps intensities that are within np.linspace(0, max_val, 7)
    """
    mod_psych_table = []
    for row in psych_table:
        # can't use simple row[0] in np.linspace due to floating point errors
        if np.any(np.isclose(np.linspace(0, 1.2, 7), row[0])):
            mod_psych_table.append(row)
        else:
            continue

    mod_psych_table = np.array(mod_psych_table)
    return mod_psych_table


def logistic_residuals(params, stim_intensities, true_reg):
    '''
    Residuals for the one slope logistic function.
    '''
    slope, bias = params
    model_reg  = slope * stim_intensities + bias
    res_reg = model_reg - true_reg
    return np.concatenate([res_reg])


def psych_fit(table):
    '''
    Fits the psychometric function given behavioral data from reg and rev trials.
    '''

    initial_guess = [4, -2] # for slope and bias

    percents = table[:, 1]

    # replace 0 with an infintesimally small number and 1 with 1-infintesimally small number
    percents[percents == 0] = np.finfo(float).eps
    percents[percents == 1] = 1 - np.finfo(float).eps
    percents_log = np.log(percents / (1 - percents))

    residual_func = logistic_residuals

    fit = least_squares(
        residual_func,
        x0 = initial_guess,
        args=(table[:, 0], percents_log),
        method='trf'
        )

    slope, bias = fit.x
    pars = {
        'slope': slope,
        'bias': bias,
    }

    return pars


def calc_psych_curve(fit_pars, max_val=1):
    '''
    Return the psychometric curve based on fit parameters.
    '''
    slope = fit_pars['slope']
    psych_curve = 100*(1/(1+np.exp(-slope*(np.linspace(0, max_val, max_val*1000))-fit_pars['bias'])))

    return psych_curve
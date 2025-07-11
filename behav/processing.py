import behav.psignifit as ps
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

import ipywidgets as widgets
from IPython.display import display, clear_output

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

    opto_data['optoStim'] = opto_data['optoStim'].replace({
        'S1-L': 'S1C2-L',
        'S1-R': 'S1C2-R'
    })

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
    index = np.argsort(psych_table[:, 0])
    psych_table = psych_table[index]

    return psych_table

def make_psych_table_psignifit(processed_data, optoStim = 'none'):
    """
    For a given optoStim, makes a psychometric table of the whisker stimulation intensity,
    the no. of lick L, and total no. of trials.
    If no optoStim provided, considers data for no optogenetic stimulation.
    """
    opto_data = processed_data[processed_data['optoStim'] == optoStim]

    opto_data['optoStim'] = opto_data['optoStim'].replace({
        'S1-L': 'S1C2-L',
        'S1-R': 'S1C2-R'
    })

    lick_L = (opto_data['trial_type'] == 3) | (opto_data['trial_type'] == 20)

    no_of_intensities = len(set(opto_data['stim_left']))

    psych_table = np.zeros((no_of_intensities, 3))

    for index in range(no_of_intensities):
        psych_table[index, 0] = opto_data['stim_left'].unique()[index]
        psych_table[index, 1] = len(opto_data[lick_L]
                                    [opto_data['stim_left'] == psych_table[index, 0]])
        psych_table[index, 2] = len(opto_data[opto_data['stim_left'] == psych_table[index, 0]])
        
    psych_table = np.array(psych_table)
    index = np.argsort(psych_table[:, 0])
    psych_table = psych_table[index]

    return psych_table

def take_needed_ints_only(psych_table, max_val = 1.2):
    """
    Given psych_table, only keeps intensities that are within np.linspace(0, max_val, 7)
    """
    mod_psych_table = []
    for row in psych_table:
        # can't use simple row[0] in np.linspace due to floating point errors
        if np.any(np.isclose(np.linspace(0, max_val, 7), row[0])):
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

    initial_guess = [5, -2] # for slope and bias

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

def fit_weibull(psych_table_ps, optoStim = 'control'):

    c = np.log(-np.log(0.05)) - np.log(-np.log(0.95))
    fit_results = {}
    baseline_trials = psych_table_ps[0, 2]
    baseline_fraction = (psych_table_ps[0, 1] / baseline_trials 
                         if baseline_trials != 0 else np.nan)

    fit_opts = {
        'sigmoidName': 'weibull',
        'expType': 'YesNo',
        'fixedPars': np.full((5, 1), np.nan)
    }
    fit_opts['fixedPars'][3] = baseline_fraction

    data_fit = psych_table_ps[1:, :]
    result = ps.psignifit(data_fit, fit_opts)
    fit_pars = result['Fit']
    threshold = ps.getThreshold(result, 0.5)[0]

    x_data = psych_table_ps[:, 0]
    y_data = (psych_table_ps[:, 1] / psych_table_ps[:, 2]) * 100
    max_intensity = np.max(x_data)
    x_vals_cond = np.linspace(0, max_intensity, 1000)
    
    exp_values = np.exp(c * (np.log(x_vals_cond) - fit_pars[0]) / fit_pars[1])
    psych_curve = 100 * ((1 - fit_pars[3] - fit_pars[2]) 
                         *
                         (1 - np.exp(np.log(0.5) * exp_values)) + fit_pars[3])
    
    fit_results = {
        # 'condition': optoStim,
        'psych_curve': psych_curve,
        'x_vals': x_vals_cond,
        'x_data': x_data,
        'y_data': y_data,
        'fit_pars': fit_pars,
        'threshold': threshold,
        'max_intensity': max_intensity
    }

    return fit_results


def make_plots(opto_data, show_data = False):

    opto_data['optoStim'] = opto_data['optoStim'].replace({
        'S1-L': 'S1C2-L',
        'S1-R': 'S1C2-R'
    })


    def plot_fit(opto_data, optoStim = 'none', max_val = 1.2, show_data = show_data):
        psych_table = make_psych_table(opto_data, optoStim = optoStim)
        psych_table = take_needed_ints_only(psych_table, max_val = max_val)
        fit_params = psych_fit(psych_table)
        if show_data == True:
            plt.scatter(psych_table[:, 0]/1.2, psych_table[:, 1]*100, c = 'r')
        plt.plot(np.linspace(0, 1, 1000), calc_psych_curve(fit_params), label = optoStim)
        return None
    
    # Get unique values from the 'optoStim' column
    unique_stims = sorted(opto_data['optoStim'].unique())
    unique_stims.remove('none')

    # Create a checkbox for each unique value
    checkboxes = {stim: widgets.Checkbox(value = False if stim != 'control' else True, description=str(stim)) 
                for stim in unique_stims}
    checkbox_container = widgets.VBox(list(checkboxes.values()))

    # Define plot update function
    def update_plot(change=None):
        clear_output(wait=True)
        display(checkbox_container)
        
        # Start the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot for each selected stim
        for stim, cb in checkboxes.items():
            if cb.value:
                plot_fit(opto_data, optoStim = stim, max_val = 1.2, show_data=show_data)

        plt.title("Psychometric curves using logistic fitting")
        plt.xlabel("Normalized whisker stimulation intensity")
        plt.ylabel("% Lick left")        
        ax.legend()
        ax.grid(True)
        plt.show()

    # Attach the update function to all checkboxes
    for cb in checkboxes.values():
        cb.observe(update_plot, 'value')

    # Initial display
    update_plot()
    return None

def make_plots_ps(opto_data_ps, show_data = False):
    
    opto_data_ps['optoStim'] = opto_data_ps['optoStim'].replace({
        'S1-L': 'S1C2-L',
        'S1-R': 'S1C2-R'
    })

    def plot_fit(opto_data_ps, optoStim = 'none', max_val = 1.2, show_data = show_data):
        
        psych_table_ps = make_psych_table_psignifit(opto_data_ps, optoStim = optoStim)
        psych_table_ps = take_needed_ints_only(psych_table_ps, max_val = max_val)
        
        curve_fit = fit_weibull(psych_table_ps, optoStim = optoStim)
        
        if show_data == True:
            plt.scatter(curve_fit['x_data']/max_val, 
                        curve_fit['y_data'], c = 'r')
        plt.plot(curve_fit['x_vals']/max_val, curve_fit['psych_curve'], 
                 label = optoStim)
        return None
    
    # Get unique values from the 'optoStim' column
    unique_stims = sorted(opto_data_ps['optoStim'].unique())
    unique_stims.remove('none')

    # Create a checkbox for each unique value
    checkboxes = {stim: widgets.Checkbox(value = False if stim != 'control' else True, description=str(stim)) 
                for stim in unique_stims}
    checkbox_container = widgets.VBox(list(checkboxes.values()))

    # Define plot update function
    def update_plot(change=None):
        clear_output(wait=True)
        display(checkbox_container)
        
        # Start the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot for each selected stim
        for stim, cb in checkboxes.items():
            if cb.value:
                plot_fit(opto_data_ps, optoStim = stim, max_val = 1.2, show_data=show_data)
        
        plt.title("Psychometric curves using Weibull fitting")
        plt.xlabel("Normalized whisker stimulation intensity")
        plt.ylabel("% Lick left")
        ax.legend()
        ax.grid(True)
        plt.show()

    # Attach the update function to all checkboxes
    for cb in checkboxes.values():
        cb.observe(update_plot, 'value')

    # Initial display
    update_plot()
    return None
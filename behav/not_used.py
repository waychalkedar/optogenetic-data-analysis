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

def find_subthreshold(df):
    """
    In order to find the sub TH values the function identifies the maximum stimulus intensity for each side (a value that occur more than 
    20 times). Using these maximum values, the function isolates bilateral trials (where both sides are stimulated) and extracts the sub
    TH intensities by selecting trials in which one side's intensity is below its maximum while the other matches its maximum. 
    Finally, it returns the modified DataFrame along with the computed sub TH values.
    """
    # Find max stimuli for each side
    df = round_stim_data(df)
    def find_valid_max(series):
        sorted_values = series[series > 0].value_counts().sort_index(ascending=False)
        valid = sorted_values[sorted_values > 20]
        return valid.index[0] if not valid.empty else "Not Found"

    max_L = find_valid_max(df['stim_left'])
    max_R = find_valid_max(df['stim_right'])

    # Find sub TH value for each side
    bilateral_trials = df[(df['stim_left'] > 0) & (df['stim_right'] > 0)]
    sub_th_L = bilateral_trials[(bilateral_trials['stim_right'] == max_R) & 
                                (bilateral_trials['stim_left'] != max_R)]['stim_left'].value_counts()
    sub_th_R = bilateral_trials[(bilateral_trials['stim_left'] == max_L) & 
                                (bilateral_trials['stim_right'] != max_L)]['stim_right'].value_counts()
    sub_th_L = sub_th_L.idxmax() if not sub_th_L.empty else "Not Found"
    sub_th_R = sub_th_R.idxmax() if not sub_th_R.empty else "Not Found"
    
    return df, sub_th_L, sub_th_R

def generate_psychometric_matrices(df, sub_th_L, sub_th_R):
    """
    This function generates four psychometric matrices per unique optoStim condition, by dividing the trial data into unilateral and bilateral 
    groups. Unilateral matrices consist of trials with stimulation on only one side (with the other at zero), including a baseline row for 
    zero intensity (0 - 0 trials). Bilateral matrices are built from trials where both sides are stimulated, with one side at the sub TH value.
    (Note: Due to an experimental error with 0 – sub TH trials, the first row for bilateral matrices is temporarily fixed at 0.2, being the 
    closest value, instead of the true sub TH values; this will be updated when more data is available). 
    For each category, trials are grouped by stimulus intensity to compute the number of correct left licks (trial_type = 3, 20) and the 
    total trial count. Each resulting matrix is structured with three columns: intensity, number of correct licks, and total trials, and is 
    stored in a dictionary keyed by (optoStim, condition) for later analysis with psignifit.
    """

    matrices = {}
    unique_optoStims = df['optoStim'].unique()
    for opto in unique_optoStims:
        df_opto = df[df['optoStim'] == opto]
        unilateral_L = df_opto[(df_opto['stim_right'] == 0) & 
                               (df_opto['stim_left'] > 0) & 
                               (df_opto['stim_left'] != sub_th_L)]
        unilateral_R = df_opto[(df_opto['stim_left'] == 0) & 
                               (df_opto['stim_right'] > 0) & 
                               (df_opto['stim_right'] != sub_th_R)]
        bilateral_L = df_opto[df_opto['stim_right'] == sub_th_R]
        bilateral_R = df_opto[df_opto['stim_left'] == sub_th_L]
        zero_intensity = df_opto[(df_opto['stim_left'] == 0) & (df_opto['stim_right'] == 0)]
        bilateral_L_first_row = df_opto[(df_opto['stim_left'] == 0) & ((df_opto['stim_right'] == 0.2) | (df_opto['stim_right'] == sub_th_R))]
        bilateral_R_first_row = df_opto[(df_opto['stim_right'] == 0) & ((df_opto['stim_left'] == 0.2) | (df_opto['stim_left'] == sub_th_L))]
        
        def create_matrix(df_category, intensity_col, first_row_df=None):
            grouped = df_category.groupby(intensity_col)['trial_type'].agg([
                ('num_lick_L', lambda x: np.sum((x == 3) | (x == 20))),
                ('num_trials', 'count')
            ]).reset_index()
            grouped = grouped.rename(columns={intensity_col: 'intensity'})
            if first_row_df is not None and not first_row_df.empty:
                first_licks = np.sum((first_row_df['trial_type'] == 3) | (first_row_df['trial_type'] == 20))
                first_row = pd.DataFrame({'intensity': [0.0], 'num_lick_L': [first_licks], 'num_trials': [len(first_row_df)]})
                grouped = pd.concat([first_row, grouped], ignore_index=True)
            elif first_row_df is None and not zero_intensity.empty:
                zero_licks = np.sum((zero_intensity['trial_type'] == 3) | (zero_intensity['trial_type'] == 20))
                zero_row = pd.DataFrame({'intensity': [0.0], 'num_lick_L': [zero_licks], 'num_trials': [len(zero_intensity)]})
                grouped = pd.concat([zero_row, grouped], ignore_index=True)
            return grouped.astype(float).to_numpy()
        
        matrices[(opto, 'Unilateral L')] = create_matrix(unilateral_L, 'stim_left')
        matrices[(opto, 'Unilateral R')] = create_matrix(unilateral_R, 'stim_right')
        matrices[(opto, f'Bilateral L (Sub-TH R = {sub_th_R})')] = create_matrix(bilateral_L, 'stim_left', bilateral_L_first_row)
        matrices[(opto, f'Bilateral R (Sub-TH L = {sub_th_L})')] = create_matrix(bilateral_R, 'stim_right', bilateral_R_first_row)
    return matrices

def filter_matrices_by_trials(matrices_dict, min_trials=10):
    """
    Filters each psychometric matrix to only include intensities with a minimum of 10 trials.
    """

    filtered_dict = {}
    for key, matrix in matrices_dict.items():
        filtered_matrix = matrix[matrix[:, 2] >= min_trials]
        filtered_dict[key] = filtered_matrix
    return filtered_dict

def filter_none_matrices(filtered_mats):
    """
    In the matrices_dict dictionary, each key is made up of two parts: the first part indicates the optoStim condition (like 'none') and 
    the second part describes the trial type (such as 'Unilateral L'). This function returns a new dictionary containing only the items 
    with 'none' as the optoStim condition.
    """

    return {k: v for k, v in filtered_mats.items() if k[0].lower() == 'none'}

def fit_psychometric_curves(none_mats):
    """
    Fits psychometric curves for each condition in the provided matrices using psignifit with a Weibull sigmoid model (Yes/No design). 
    For each condition, the function:
      - Fixes the lower asymptote (baseline) using fit options.
      - Retrieves fit parameters (fit_pars) that describe the curve, where:
            fit_pars[0]: Threshold (intensity at 50% performance)
            fit_pars[1]: Slope (steepness of the curve)
            fit_pars[2]: Lapse rate (upper asymptote error)
            fit_pars[3]: Guess rate (baseline performance, fixed via fit_opts)
            fit_pars[4]: Shape parameter (curvature adjustment)
      - Uses a constant (c) computed from the 0.05 and 0.95 percentiles to transform 
        the stimulus intensities and generate an exponential term.
      - Calculates the final psychometric curve (psych_curve) in percentage.
    
    Returns a dictionary of fit results for each condition. Each result includes:
      - 'psych_curve': the computed psychometric curve,
      - 'x_vals': the range of stimulus intensities used for plotting,
      - 'x_data' and 'y_data': the original data points,
      - 'fit_pars': the parameters from the psignifit fitting,
      - 'threshold': the computed threshold (the intensity at 50% performance),
      - 'max_intensity': the highest intensity value in the data.
    """

    c = np.log(-np.log(0.05)) - np.log(-np.log(0.95))
    fit_results = {}
    for key, matrix in none_mats.items():
        condition = key[1]
        baseline_trials = matrix[0, 2]
        baseline_prop = matrix[0, 1] / baseline_trials if baseline_trials != 0 else np.nan

        fit_opts = {
            'sigmoidName': 'weibull',
            'expType': 'YesNo',
            'fixedPars': np.full((5, 1), np.nan)
        }
        fit_opts['fixedPars'][3] = baseline_prop

        data_fit = matrix[1:, :]
        result = ps.psignifit(data_fit, fit_opts)
        fit_pars = result['Fit']
        threshold = ps.getThreshold(result, 0.5)[0]

        x_data = matrix[:, 0]
        y_data = (matrix[:, 1] / matrix[:, 2]) * 100
        max_intensity = np.max(x_data)
        total_trials = int(np.sum(matrix[:, 2]))   # <-- new line
        x_vals_cond = np.linspace(0, max_intensity, 1000)

        exp_values = np.exp(c * (np.log(x_vals_cond) - fit_pars[0]) / fit_pars[1])
        psych_curve = 100 * ((1 - fit_pars[3] - fit_pars[2]) * (1 - np.exp(np.log(0.5) * exp_values)) + fit_pars[3])

        fit_results[condition] = {
            'condition': condition,
            'psych_curve': psych_curve,
            'x_vals': x_vals_cond,
            'x_data': x_data,
            'y_data': y_data,
            'fit_pars': fit_pars,
            'threshold': threshold,
            'max_intensity': max_intensity,
            'total_trials': total_trials   

        }
    return fit_results

def plot_fit_results(fit_results, mouse_id, day_range, opto_name, sub_th_L, sub_th_R):
    """
    Plots the fitted psychometric curves using the fit_results. Creates a figure for each optoStim, with two subplots, 1 for each side of
    stimulus. Each subplot shows both unilateral and bilateral conditions for the side, including the scatter of data points and the TH 
    value calculated by psignfit. 
    The legend shows the Weibull fit (lines) and threshold (vertical lines). An additional text box displays the trials count for each 
    condition.
    """

    indiv_UL = fit_results.get('Unilateral L', None)
    indiv_UR = fit_results.get('Unilateral R', None)
    indiv_BL = None
    indiv_BR = None

    for cond, res in fit_results.items():
        cond_lower = cond.lower()
        if cond_lower.startswith('bilateral l'):
            indiv_BL = res
        elif cond_lower.startswith('bilateral r'):
            indiv_BR = res

    if any(x is None for x in [indiv_UL, indiv_BL, indiv_UR, indiv_BR]):
        print("Warning: Some expected conditions were not found.")

    color_mapping = {
        'Unilateral L': '#D95319',
        'Bilateral L': '#570903',
        'Unilateral R': '#0072BD',
        'Bilateral R': '#010036'
    }

    fixed_stimuli = np.arange(0, 1.2, 0.2)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle(f"Psychometric Curve for Mouse {mouse_id}, {day_range}\nOptoStim: {opto_name}",
                 fontsize=18, fontweight='bold')

    # Left subplot: L Unilateral & Bilateral
    ax = axes[0]
    max_left = 0
    if indiv_UL:
        ax.plot(indiv_UL['x_vals'], indiv_UL['psych_curve'],
                color=color_mapping['Unilateral L'],
                linewidth=2, alpha=0.5, label='Unilateral L (U-L)')
        filtered_indices_UL = np.isin(indiv_UL['x_data'], fixed_stimuli)
        ax.scatter(indiv_UL['x_data'][filtered_indices_UL], indiv_UL['y_data'][filtered_indices_UL],
                   s=60, color=color_mapping['Unilateral L'], edgecolor='black', alpha=0.3)
        ax.axvline(indiv_UL['threshold'], color=color_mapping['Unilateral L'],
                   linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'TH U-L: {indiv_UL["threshold"]:.2f}')
        max_left = max(max_left, indiv_UL['max_intensity'])
    if indiv_BL:
        ax.plot(indiv_BL['x_vals'], indiv_BL['psych_curve'],
                color=color_mapping['Bilateral L'],
                linewidth=2, alpha=0.8, label='Bilateral L')
        filtered_indices_BL = np.isin(indiv_BL['x_data'], fixed_stimuli)
        ax.scatter(indiv_BL['x_data'][filtered_indices_BL], indiv_BL['y_data'][filtered_indices_BL],
                   s=60, color=color_mapping['Bilateral L'], edgecolor='black', alpha=0.6)
        ax.axvline(indiv_BL['threshold'], color=color_mapping['Bilateral L'],
                   linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'TH B-L: {indiv_BL["threshold"]:.2f}')
        max_left = max(max_left, indiv_BL['max_intensity'])
    ax.set_title(f"L Unilateral & Bilateral (R constant sub TH = {sub_th_R})")
    ax.set_xlabel('L Stimulus Intensity')
    ax.set_ylabel('Lick Left Proportion (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max_left)
    ax.legend(fontsize=8, loc='upper left')
    
    # Text box for trials count in the left subplot
    uni_left_trials = indiv_UL["total_trials"] if indiv_UL else 0
    bi_left_trials = indiv_BL["total_trials"] if indiv_BL else 0
    left_text = f"Trials Count:\nUnilateral: {uni_left_trials}\nBilateral: {bi_left_trials}"
    ax.text(0.02, 0.69, left_text, transform=ax.transAxes, fontsize=8,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Right subplot: R Unilateral & Bilateral 
    ax = axes[1]
    max_right = 0
    if indiv_UR:
        ax.plot(indiv_UR['x_vals'], indiv_UR['psych_curve'],
                color=color_mapping['Unilateral R'],
                linewidth=2, alpha=0.5, label='Unilateral R (U-R)')
        filtered_indices_UR = np.isin(indiv_UR['x_data'], fixed_stimuli)
        ax.scatter(indiv_UR['x_data'][filtered_indices_UR], indiv_UR['y_data'][filtered_indices_UR],
                   s=60, color=color_mapping['Unilateral R'], edgecolor='black', alpha=0.3)
        ax.axvline(indiv_UR['threshold'], color=color_mapping['Unilateral R'],
                   linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'TH U-R: {indiv_UR["threshold"]:.2f}')
        max_right = max(max_right, indiv_UR['max_intensity'])
    if indiv_BR:
        ax.plot(indiv_BR['x_vals'], indiv_BR['psych_curve'],
                color=color_mapping['Bilateral R'],
                linewidth=2, alpha=0.8, label='Bilateral R')
        filtered_indices_BR = np.isin(indiv_BR['x_data'], fixed_stimuli)
        ax.scatter(indiv_BR['x_data'][filtered_indices_BR], indiv_BR['y_data'][filtered_indices_BR],
                   s=60, color=color_mapping['Bilateral R'], edgecolor='black', alpha=0.6)
        ax.axvline(indiv_BR['threshold'], color=color_mapping['Bilateral R'],
                   linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'TH B-R: {indiv_BR["threshold"]:.2f}')
        max_right = max(max_right, indiv_BR['max_intensity'])
    ax.set_title(f"R Unilateral & Bilateral (L constant sub TH = {sub_th_L})")
    ax.set_xlabel('R Stimulus Intensity')
    ax.set_ylabel('Lick Left Proportion (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max_right)
    ax.legend(fontsize=8, loc='upper left')
    
    # Text box for trials count in the right subplot
    uni_right_trials = indiv_UR["total_trials"] if indiv_UR else 0
    bi_right_trials = indiv_BR["total_trials"] if indiv_BR else 0
    right_text = f"Trials Count:\nUnilateral: {uni_right_trials}\nBilateral: {bi_right_trials}"
    ax.text(0.02, 0.69, right_text, transform=ax.transAxes, fontsize=8,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def run_processing(params):
    """
    This functions Runs the full data processing pipeline using user - provided parameters.
    It loads and processes the .mat files, generates psychometric matrices, fits the psychometric curves,
    and displays the results. If an error occurs (for example, not enough data for psignifit), it shows an error message
    and restarts the GUI for new input.
    """

    try:
        mouse_id = params['mouse_id']
        day_start = params['day_start']
        day_end = params['day_end']
        base_path = params['base_path']
        
        loaded_data = load_mat_files(base_path, mouse_id, day_start, day_end)
        if not loaded_data:
            print("No .mat files loaded. Exiting.")
            return

        all_trials_df = extract_data_to_df(loaded_data)
        if all_trials_df.empty:
            print("No valid trial data found. Exiting.")
            return

        all_trials_df, sub_th_L, sub_th_R = round_stim_data(all_trials_df)
        
        psychometric_matrices = generate_psychometric_matrices(all_trials_df, sub_th_L, sub_th_R)
        filtered_mats = filter_matrices_by_trials(psychometric_matrices, min_trials=10)
        none_mats = filter_none_matrices(filtered_mats)
        if not none_mats:
            print("No matrices found for optoStim 'none'.")
            return
        
        day_range = f"Day {day_start}-{day_end}"
        opto_name = list(none_mats.keys())[0][0]
        fit_results = fit_psychometric_curves(none_mats)
        plot_fit_results(fit_results, mouse_id, day_range, opto_name, sub_th_L, sub_th_R)

    except AssertionError as e:
        import tkinter.messagebox as mb
        mb.showerror("Analysis Error", "Insufficient data to preform analysis, load more files")
        run_gui()

def select_base_folder():
    """ 
    First window: Select the base folder through browsing. 
    """

    root = tk.Tk()
    root.title("Select Base Folder")
    root.geometry("500x150")
    style = ttk.Style(root)
    style.configure("TLabel", font=("Helvetica", 12))
    style.configure("TButton", font=("Helvetica", 12))
    
    folder_var = tk.StringVar()
    
    ttk.Label(root, text="Please select the base folder for your .mat files").pack(pady=10)
    
    frame = ttk.Frame(root)
    frame.pack(pady=5)
    entry = ttk.Entry(frame, textvariable=folder_var, width=40)
    entry.pack(side=tk.LEFT, padx=5)
    
    def browse():
        folder = filedialog.askdirectory(title="Select the base folder for your .mat files")
        folder_var.set(folder)
    
    ttk.Button(frame, text="Browse", command=browse).pack(side=tk.LEFT, padx=5)
    
    def on_continue():
        if folder_var.get():
            root.destroy()
        else:
            print("Please select a folder.")
    
    ttk.Button(root, text="Continue", command=on_continue).pack(pady=10)
    root.mainloop()
    return folder_var.get()

def input_parameters():
    """ 
    Second window: Insert Mouse ID, Day Start, and Day End. 
    """

    params = {}
    root = tk.Tk()
    root.title("Enter Parameters")
    root.geometry("400x220")
    style = ttk.Style(root)
    style.configure("TLabel", font=("Helvetica", 12))
    style.configure("TButton", font=("Helvetica", 12))
    
    ttk.Label(root, text="Enter the experiment parameters below:").grid(row=0, column=0, columnspan=2, pady=10)
    
    ttk.Label(root, text="Mouse ID:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    mouse_entry = ttk.Entry(root)
    mouse_entry.grid(row=1, column=1, padx=10, pady=5)
    
    ttk.Label(root, text="Day Start:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    day_start_entry = ttk.Entry(root)
    day_start_entry.grid(row=2, column=1, padx=10, pady=5)
    
    ttk.Label(root, text="Day End:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    day_end_entry = ttk.Entry(root)
    day_end_entry.grid(row=3, column=1, padx=10, pady=5)
    
    error_label = ttk.Label(root, text="", foreground="red")
    error_label.grid(row=4, column=0, columnspan=2)
    
    def on_continue():
        try:
            params['mouse_id'] = mouse_entry.get().strip()
            params['day_start'] = int(day_start_entry.get().strip())
            params['day_end'] = int(day_end_entry.get().strip())
            root.destroy()
        except ValueError:
            error_label.config(text="Please enter valid numbers for Day Start and Day End.")
    
    ttk.Button(root, text="Continue", command=on_continue).grid(row=5, column=0, columnspan=2, pady=10)
    root.mainloop()
    return params

def run_gui():
    """ 
    Runs the two-step GUI and then the processing pipeline. 
    """

    base_folder = select_base_folder()
    if not base_folder:
        print("No folder selected. Exiting.")
        return
    params = input_parameters()
    params['base_path'] = base_folder
    run_processing(params)
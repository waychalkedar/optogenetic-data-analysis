import os
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat
import warnings
warnings.simplefilter("ignore")

def load_mat_files(base_path, mouse_id, day_start, day_end):
    """
    Loads .mat files for the specified mouse and day range.
    If the base_path already ends with the mouse_id, it uses that folder directly;
    otherwise, it appends the mouse_id to the base path.
    Returns a dictionary mapping file paths to loaded data.
    """

    if os.path.basename(os.path.normpath(base_path)) == mouse_id:
        subject_folder = base_path
    else:
        subject_folder = os.path.join(base_path, mouse_id)
    
    if not os.path.isdir(subject_folder):
        print("Error: Subject folder does not exist. Check the mouse_id and base path.")
        return {}
    mat_files = {}
    for day in range(day_start, day_end + 1):
        pattern = os.path.join(subject_folder, f"*Day {day},*.mat")
        for file in glob.glob(pattern):
            mat_files[file] = loadmat(file)
    return mat_files

def extract_data_to_df(mat_files):
    """
    The mat file includes a variable named "TrainingTraces" that holds several sub-variables.
    This function extracts the essential variables for analysis (trial types, stimulus strengths, and optoStim)
    from the loaded .mat files and returns a combined DataFrame of all the trials.
    During the extraction process, it reformats the optoStim variable (originally stored as a single string with trial separated by 
    semicolon) by splitting it into a list of individual trial values. It then verifies that the number of optoStim entries matches the 
    expected trial count; if not, the file is skipped.
    """

    extracted_data = []
    for file, data in mat_files.items():
        training_traces = data["TrainingTraces"]
        trial_types = training_traces['trialTypes'][0, 0].flatten().astype(int)
        stim_strengths = training_traces['stimStrengths'][0, 0]
         
        # Fix optoStim shape:
        optoStim_str = training_traces['optoStim'][0, 0].item()
        optoStim_list = [s.strip() for s in optoStim_str.split(';') if s.strip()]
        
        n_trials = trial_types.shape[0]
        if not (len(optoStim_list) == n_trials and stim_strengths.shape[0] == n_trials):
            continue
        
        extracted_data.append(pd.DataFrame({
            'trial_type': trial_types,
            'stim_left': stim_strengths[:, 0],
            'stim_right': stim_strengths[:, 1],
            'optoStim': optoStim_list,
            'file': [file] * n_trials
        }))
        
    return pd.concat(extracted_data, ignore_index=True) if extracted_data else pd.DataFrame()

def remove_right_stim(df):
    return df.drop('stim_right', axis = 1)

def round_stim_data(df):
    """
    This function first rounds up intensities ending in .99 or .49 to ensure consistent processing and then it locates the sub threshold 
    (sub TH) values. The rounding up is done on these 2 intensities specifically due to the sub-threshold calculation being sensitive to 
    small differences (around 0.05).
    """
    for col in ['stim_left', 'stim_right']:
        mask = np.isclose(df[col] % 1, 0.99, atol=1e-6) | np.isclose(df[col] % 1, 0.49, atol=1e-6)
        df.loc[mask, col] += 0.01
        df[col] = np.round(df[col], 2)
    return df

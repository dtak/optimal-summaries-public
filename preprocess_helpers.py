import numpy as np
import pandas as pd
import pickle

changing_vars = [
 'dbp',
 'fio2',
 'GCS',
 'hr',
 'map',
 'sbp',
 'spontaneousrr',
 'spo2',
 'temp',
 'urine',
 'bun',
 'magnesium',
 'platelets',
 'sodium',
 'alt',
 'hct',
 'po2',
 'ast',
 'potassium',
 'wbc',
 'bicarbonate',
 'creatinine',
 'lactate',
 'pco2',
 'glucose',
 'inr',
 'hgb',
 'bilirubin_total']

def preprocess_MIMIC_data(x_filepath, y_filepath, y_2d=True):
    """
    Returns pre-processed numpy arrays X and y after re-ordering columns
    from pickled Dataframe arrays x_filepath and y_filepath.
    
    Current script only includes first 24 time-steps.
    
    Args:
        x_filepath: file path to pickled array of covariate data.
        y_filepath: file path to pickled array of target data.
        y_2d: boolean equal to true if targets should be returned as a 2D array.
    """
    
    X = pickle.load( open(x_filepath, "rb" ) )
    Y = pickle.load( open(y_filepath, "rb" ) )
    
    # Drop all the GCS variables that are not the sum.
    GCS_other = ['GCS_eye',
           'GCS_eye_ind', 'GCS_motor', 'GCS_motor_ind', 'GCS_verbal',
           'GCS_verbal_ind']

    data_cols = changing_vars.copy()
    
    # The next cols are the same data missingness indicators
    for c in changing_vars:
        data_cols.append(c + '_ind')
        
    for c in X[0].columns:
        if c not in data_cols and c not in GCS_other:
            data_cols.append(c)
            
    X_np = []
    Y_new = []

    # X_np will have the first 6 hours of data for all patient series.

    for i in range(len(X)):
        pat = X[i].reindex(columns=data_cols)
        y = Y[i]
        pat_arr = np.array(pat)

        if pat_arr.shape[0] > 6:
            # use the first 6 hours of data
            X_np.append(np.array(pat)[:6])
            Y_new.append(y[0])

    Y = Y_new.copy()
    X_np = np.array(X_np)
    
    if not y_2d:
        return X_np, y
    
    # create Y_logits
    Y_logits = []

    for y in Y:
        if y == 0:
            Y_logits.append([1, 0])
        else:
            Y_logits.append([0, 1])
            
    return X_np, Y_logits, changing_vars, data_cols

# Load mean and variance arrays.
mu_dict = pickle.load( open( "data/mu_dict_vasopressor_LOS_6_600.p", "rb" ) )
var_dict = pickle.load( open( "data/var_dict_vasopressor_LOS_6_600.p", "rb" ) ) 

def inverse_feature_preprocessing(x, feat_name):
    """
    Returns the feature x when transformed back into its original 
    representation space by "undo"ing its pre-processing.
    
    Args:
        x (float): the quantity to be transformed
        feat_name (str): the name of the feature to be transformed
    """
    
    mu = mu_dict[feat_name]
    var = var_dict[feat_name]
    
    return var * x + mu
    
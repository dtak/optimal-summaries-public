import os
import sys
import argparse
import time
import csv
import random

import numpy as np
import pandas as pd
import pickle
import torch
# import higher

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from weights_parser import WeightsParser

from models import LogisticRegressionWithSummaries_Wrapper

from custom_losses import custom_bce_horseshoe
from param_initializations import *

from preprocess_helpers import preprocess_MIMIC_data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits

X_np, Y_logits, changing_vars, data_cols = preprocess_MIMIC_data('data/X_vasopressor_LOS_6_600.p', 'data/y_vasopressor_LOS_6_600.p')

parser = argparse.ArgumentParser()

parser.add_argument('--split_random_state', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--init_cutoffs', type=str, default='init_cutoffs_to_zero')
parser.add_argument('--cutoff_times_init_values_filepath', type=str, default='')
parser.add_argument('--init_thresholds', type=str, default='init_rand')
parser.add_argument('--cutoff_times_temperature', type=float, default=1.0)
parser.add_argument('--thresholds_temperature', type=float, default=0.1)
parser.add_argument('--ever_measured_temperature', type=float, default=0.1)
parser.add_argument('--switch_temperature', type=float, default=0.1)
parser.add_argument('--top_k',type=str, default='')

parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--tau', type=float, default=10.)
parser.add_argument('--opt_lr', type=float, default=1e-4)
parser.add_argument('--opt_weight_decay', type=float, default=0.)

parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--model_output_name', type=str, default='')
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--save_every', type=int, default=100)


FLAGS = parser.parse_args()

# Set up path to save model artifacts, possibly suffixed with an experiment ID
path = FLAGS.output_dir or f"/tmp/{int(time.time())}"
os.system('mkdir -p ' + path)

device = torch.device("cuda:0")  # Uncomment this to run on GPU
torch.cuda.get_device_name(0)
torch.cuda.is_available()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# train-test-split
torch.set_printoptions(sci_mode=False)
X_train, X_test, y_train, y_test = train_test_split(X_np, Y_logits, test_size = 0.15, random_state = FLAGS.split_random_state, stratify = Y_logits)

# train-val split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = FLAGS.split_random_state, stratify = y_train)

from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

def tensor_wrap(x, klass=torch.Tensor):
    return x if 'torch' in str(type(x)) else klass(x)

set_seed(FLAGS.split_random_state)

X_pt = Variable(tensor_wrap(X_np)).cuda()

pos_prop = np.mean(np.array(Y_logits)[:, 1])

p_weight = torch.Tensor([1 / (1 - pos_prop), 1 / pos_prop]).cuda()

X_train_pt = Variable(tensor_wrap(X_train)).cuda()
y_train_pt = Variable(tensor_wrap(y_train, torch.FloatTensor)).cuda()

X_val_pt = Variable(tensor_wrap(X_val)).cuda()
y_val_pt = Variable(tensor_wrap(y_val, torch.FloatTensor)).cuda()

X_test_pt = Variable(tensor_wrap(X_test)).cuda()
y_test_pt = Variable(tensor_wrap(y_test, torch.FloatTensor)).cuda()

batch_size = FLAGS.batch_size

train_dataset = TensorDataset(X_train_pt, y_train_pt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = TensorDataset(X_val_pt, y_val_pt)
val_loader = DataLoader(val_dataset, batch_size = X_val_pt.shape[0], shuffle=True, num_workers=0)

test_dataset = TensorDataset(X_test_pt, y_test_pt)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

input_dim = X_np[0].shape[1]
changing_dim = len(changing_vars)

cutoff_init_fn = init_cutoffs_to_zero

if FLAGS.init_cutoffs == 'init_cutoffs_to_twelve':
    cutoff_init_fn = init_cutoffs_to_twelve

elif FLAGS.init_cutoffs == 'init_cutoffs_to_twentyfour':
    cutoff_init_fn = init_cutoffs_to_twentyfour
    
elif FLAGS.init_cutoffs == 'init_cutoffs_randomly':
    cutoff_init_fn = init_cutoffs_randomly
    
lower_thresh_init_fn = init_rand_lower_thresholds
upper_thresh_init_fn = init_rand_upper_thresholds

if FLAGS.init_thresholds == 'zeros':
    lower_thresh_init_fn = init_zeros
    upper_thresh_init_fn = init_zeros
    
    
cutoff_times_init_values = None
if len(FLAGS.cutoff_times_init_values_filepath) > 0:
    # Load the numpy array from its filepath.
    cutoff_times_init_values = pickle.load( open( FLAGS.cutoff_times_init_values_filepath, "rb" ) )

logreg = LogisticRegressionWithSummaries_Wrapper(input_dim, 
                                                 changing_dim, 
                                                 9,
                                                 True,
                                                 cutoff_init_fn, 
                                                 lower_thresh_init_fn, 
                                                 upper_thresh_init_fn,
                                                 cutoff_times_temperature=FLAGS.cutoff_times_temperature,
                                                 cutoff_times_init_values=cutoff_times_init_values,
                                                 alpha = FLAGS.alpha,
                                                 tau = FLAGS.tau,
                                                 opt_lr = FLAGS.opt_lr,
                                                 opt_weight_decay = FLAGS.opt_weight_decay,
                                                 # top_k = FLAGS.top_k
                                                )
logreg.cuda()

set_seed(FLAGS.split_random_state)

logreg.fit(train_loader, val_loader, p_weight,
         save_model_path = FLAGS.model_output_name,
         epochs=FLAGS.num_epochs,
         save_every_n_epochs=FLAGS.save_every)

# get AUC
y_hat_test = []
for pat in X_test:
    # batch size of 1
    x = tensor_wrap([pat]).cuda()
    y_hat_test.append(logreg.model.sigmoid(logreg.model.forward(x))[:,1].item())
score = roc_auc_score(np.array(y_test)[:, 1], y_hat_test)

# write results to csv
filename = "vasopressor_baseline_gridsearch_r{}".format(FLAGS.split_random_state)
dir_path = '/n/home07/carissawu/optimal-summaries/vasopressor/models/LOS-6-600/baseline'
with open('{file_path}.csv'.format(file_path=os.path.join(dir_path, filename)), 'a+') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow([FLAGS.alpha, FLAGS.tau, FLAGS.opt_lr, FLAGS.opt_weight_decay,score]) 

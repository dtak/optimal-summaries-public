#!/usr/bin/env python
# coding: utf-8

# This script trains the LSTM using the RC GPU.


import os
import sys
import argparse
import time
import csv


import numpy as np
import pandas as pd
import pickle
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from time import sleep

import matplotlib.pyplot as plt

from preprocess_helpers import preprocess_MIMIC_data
from models import LSTM_NextState, LSTM_Baseline
from custom_losses import LSTM_compound_loss
from param_initializations import *



parser = argparse.ArgumentParser()

parser.add_argument('--split_random_state', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lstm_hidden_dim', type=int, default=32)
parser.add_argument('--hidden_next_state_dim', type=int, default=256)
parser.add_argument('--hidden_outcome_dim', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--beta', type=float, default=10.)
parser.add_argument('--opt_lr', type=float, default=1e-4)
parser.add_argument('--opt_weight_decay', type=float, default=0.)
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--model_output_name', type=str, default='')
parser.add_argument('--num_epochs', type=int, default=15000)
parser.add_argument('--save_every', type=int, default=1000)


FLAGS = parser.parse_args()


# Set up path to save model artifacts, possibly suffixed with an experiment ID
path = FLAGS.output_dir or f"/tmp/{int(time.time())}"
os.system('mkdir -p ' + path)

device = torch.device("cuda:0")  # Uncomment this to run on GPU
torch.cuda.get_device_name(0)
torch.cuda.is_available()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

X_np, Y_logits, changing_vars, data_cols = preprocess_MIMIC_data('data/X_vasopressor_LOS_6_600.p', 'data/y_vasopressor_LOS_6_600.p')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits

# Initialize LSTM

# Create X_t and X_{t + 1}.
# X_t will be 6 dimensions, X_{t + 1} will be 6 dimensions.

X_t_0 = X_np[:, :6, :]
X_t_1 = X_np[:, 1:, :]

# Do Training

changing_dim = len(changing_vars)
input_dim = len(data_cols)

pos_prop = np.mean(np.array(Y_logits)[:, 1])

p_weight = torch.Tensor([1 / (1 - pos_prop), 1 / pos_prop]).cuda()

# train-test-split
torch.set_printoptions(sci_mode=False)
X_train, X_test, y_train, y_test = train_test_split(X_np, Y_logits, test_size = 0.15, random_state = FLAGS.split_random_state)
# train-val split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = FLAGS.split_random_state)

from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

def tensor_wrap(x, klass=torch.Tensor):
    return x if 'torch' in str(type(x)) else klass(x)

set_seed(FLAGS.split_random_state)

X_pt = Variable(tensor_wrap(X_np)).cuda()


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

set_seed(FLAGS.split_random_state)

lstm = LSTM_Baseline(input_dim = input_dim,
                     changing_dim = changing_dim,
                     lstm_hidden_dim = FLAGS.lstm_hidden_dim,
                     hidden_next_state_dim = FLAGS.hidden_next_state_dim,
                     hidden_outcome_dim = FLAGS.hidden_outcome_dim,
                     alpha = FLAGS.alpha,
                     beta = FLAGS.beta,
                     opt_lr = FLAGS.opt_lr,
                     opt_weight_decay = FLAGS.opt_weight_decay)
lstm.cuda()


set_seed(FLAGS.split_random_state)

lstm.fit(train_dataset, val_dataset, p_weight,
         save_model_path = FLAGS.model_output_name,
         epochs=FLAGS.num_epochs,
         save_every_n_epochs=FLAGS.save_every)

y_hat_test = []

for pat in X_test:
    # batch size of 1
    x = tensor_wrap([pat]).cuda()
    # Forward pass.
    X_t1_pred, y_pred, _, _ = lstm.model(x)

    y_hat_test.append(lstm.model.sigmoid(y_pred)[:,1].item())

score = roc_auc_score(np.array(y_test)[:, 1], y_hat_test)

# write results to csv
filename = "vasopressor_lstm_gridsearch_r{}".format(FLAGS.split_random_state)
dir_path = '/n/home07/carissawu/optimal-summaries/vasopressor/models/LOS-6-600/lstm'
with open('{file_path}.csv'.format(file_path=os.path.join(dir_path, filename)), 'a+') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow([FLAGS.lstm_hidden_dim, FLAGS.hidden_next_state_dim, FLAGS.hidden_outcome_dim, FLAGS.alpha, FLAGS.beta,FLAGS.opt_lr,FLAGS.opt_weight_decay,score]) 


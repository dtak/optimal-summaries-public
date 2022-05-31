import os
import sys
import argparse
import time
import csv

import numpy as np
import pandas as pd
import pickle
import torch
import random

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
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

def tensor_wrap(x, klass=torch.Tensor):
    return x if 'torch' in str(type(x)) else klass(x)
        
def getAUC(model, X_test, y_test):
    # get results of forward, do AUROC
    y_hat_test = []
    for pat in X_test:
        # batch size of 1
        x = tensor_wrap([pat]).cuda()
        y_hat_test.append(model.sigmoid(model.forward(x))[:,1].item())
    score = roc_auc_score(np.array(y_test)[:, 1], y_hat_test)
    
    return score

parser = argparse.ArgumentParser()
parser.add_argument('--split_random_state', type=int, default=0)
FLAGS = parser.parse_args()

device = torch.device("cuda:0")  # Uncomment this to run on GPU
torch.cuda.get_device_name(0)
torch.cuda.is_available()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# prep data
X_np, Y_logits, changing_vars, data_cols = preprocess_MIMIC_data('data/X_vasopressor_LOS_6_600.p', 'data/y_vasopressor_LOS_6_600.p')
device = torch.device("cuda:0")  # Uncomment this to run on GPU
torch.cuda.get_device_name(0)
torch.cuda.is_available()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# begin experiment
k=40

r = FLAGS.split_random_state
# prep data
# train-test-split
torch.set_printoptions(sci_mode=False)
X_train, X_test, y_train, y_test = train_test_split(X_np, Y_logits, test_size = 0.15, random_state = r, stratify = Y_logits)

# train-val split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = r, stratify = y_train)

X_pt = Variable(tensor_wrap(X_np)).cuda()

pos_prop = np.mean(np.array(Y_logits)[:, 1])

p_weight = torch.Tensor([1 / (1 - pos_prop), 1 / pos_prop]).cuda()

X_train_pt = Variable(tensor_wrap(X_train)).cuda()
y_train_pt = Variable(tensor_wrap(y_train, torch.FloatTensor)).cuda()

X_val_pt = Variable(tensor_wrap(X_val)).cuda()
y_val_pt = Variable(tensor_wrap(y_val, torch.FloatTensor)).cuda()

X_test_pt = Variable(tensor_wrap(X_test)).cuda()
y_test_pt = Variable(tensor_wrap(y_test, torch.FloatTensor)).cuda()

batch_size = 256

train_dataset = TensorDataset(X_train_pt, y_train_pt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = TensorDataset(X_val_pt, y_val_pt)
val_loader = DataLoader(val_dataset, batch_size = X_val_pt.shape[0], shuffle=True, num_workers=0)

test_dataset = TensorDataset(X_test_pt, y_test_pt)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

input_dim = X_np[0].shape[1]
changing_dim = len(changing_vars)

topkinds = []
with open('models/LOS-6-600/baseline/topkindsbaseline_r{}.csv'.format(r), mode ='r')as file:
    # reading the CSV file
    csvFile = csv.reader(file)
    for row in csvFile:
        topkinds.append(int(row[0]))

file = open('./models/LOS-6-600/baseline/vasopressor_baseline_gridsearch_r{}.csv'.format(r))
csvreader = csv.reader(file)
header = []
header = next(csvreader)
baseline_rows = []
for row in csvreader:
    baseline_rows.append(np.array(row).astype(float))
baseline_rows = np.array(baseline_rows)
sorted_baseline_rows = baseline_rows[np.argsort(baseline_rows[:,4])]
row=sorted_baseline_rows[-1:][0]
row=[int(el) if el >= 1 else el for el in row]
row=[0 if el == 0 else el for el in row]

logreg = LogisticRegressionWithSummaries_Wrapper(input_dim, 
                                        changing_dim, 
                                        9,        
                                        True,
                                        init_cutoffs_randomly, 
                                        init_rand_lower_thresholds, 
                                        init_rand_upper_thresholds,
                                        cutoff_times_temperature=0.1,
                                        alpha = row[0],
                                        tau = row[1],
                                        opt_lr = row[2],
                                        opt_weight_decay = row[3])
logreg.cuda()
set_seed(r)
logreg.fit(train_loader, val_loader, p_weight, 
                     save_model_path = "./models/LOS-6-600/baseline/baseline_r{}_alpha_{}_tau_{}_optlr_{}_optwd_{}.pt".format(r,row[0],row[1],row[2],row[3]), 
                     epochs=10, 
                     save_every_n_epochs=10)

filename = "./models/LOS-6-600/baseline/vasopressor_baseline_r{}_topkinds_weights.csv".format(r)
with open(filename, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["AUC", "ind #"])

condition = torch.zeros(logreg.model.linear.weight.shape, dtype=torch.bool).cuda()
aucs = []
for ind in topkinds:
    condition[:,ind]=True
    temp = torch.nn.Parameter(logreg.model.linear.weight.clone().detach())
    logreg.model.linear.weight = torch.nn.Parameter(logreg.model.linear.weight.where(condition, torch.tensor(0.0).cuda()))

    # get AUC with added feature
    curr_auc = getAUC(logreg.model,X_test,y_test)

    with open(filename, 'a+') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)
        # writing the data rows 
        csvwriter.writerow([curr_auc,ind])

    logreg.model.linear.weight = temp

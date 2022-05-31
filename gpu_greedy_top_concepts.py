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

from models import LogisticRegressionWithSummariesAndBottleneck_Wrapper

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

# train-test-split
torch.set_printoptions(sci_mode=False)
X_train, X_test, y_train, y_test = train_test_split(X_np, Y_logits, test_size = 0.15, random_state = FLAGS.split_random_state, stratify = Y_logits)

# train-val split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = FLAGS.split_random_state, stratify = y_train)

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

# get top features for a set number of concepts
topkinds = []
with open('models/LOS-6-600/cos-sim/top-k/topkindsr{}c8.csv'.format(FLAGS.split_random_state), mode ='r')as file:
    # reading the CSV file
    csvFile = csv.reader(file)
    for row in csvFile:
        topkinds.append(np.array(list(map(int, row))))

# run experiment
num_concepts = 8
file = open('./models/LOS-6-600/cos-sim/vasopressor_bottleneck_r{}_c{}_gridsearch.csv'.format(FLAGS.split_random_state,num_concepts))
csvreader = csv.reader(file)
header = next(csvreader)
bottleneck_row = []
for row in csvreader:
    if row[3]=="0.001" and row[4]=="0.01":
        bottleneck_row = np.array(row).astype(float)  
# format hyperparameters for csv reader
row=[int(el) if el >= 1 else el for el in bottleneck_row]
row=[0 if el == 0 else el for el in bottleneck_row]

set_seed(FLAGS.split_random_state)
logregbottleneck = LogisticRegressionWithSummariesAndBottleneck_Wrapper(input_dim, 
                                                                            changing_dim,
                                                                            9,                     
                                                                            num_concepts,
                                                                            True,
                                                                            init_cutoffs_to_zero, 
                                                                            init_rand_lower_thresholds, 
                                                                            init_rand_upper_thresholds,
                                                                            cutoff_times_temperature=1.0,
                                                                            cutoff_times_init_values=None,
                                                                            opt_lr = row[1],
                                                                            opt_weight_decay = row[2],
                                                                            l1_lambda=row[3],
                                                                            cos_sim_lambda = row[4]
                                                                            )
logregbottleneck.cuda()
set_seed(FLAGS.split_random_state)
logregbottleneck.fit(train_loader, val_loader, p_weight, 
                     save_model_path = "./models/LOS-6-600/cos-sim/bottleneck_r{}_c{}_optlr_{}_optwd_{}_l1_lambda_{}_cossim_lambda_{}.pt".format(FLAGS.split_random_state,int(row[0]),row[1],row[2],row[3],row[4]), 
                     epochs=10, 
                     save_every_n_epochs=10)

# perform greedy selection
condition = torch.zeros(logregbottleneck.model.bottleneck.weight.shape, dtype=torch.bool).cuda()
best_aucs = []
best_auc_inds = []
best_auc_concepts = []
for i in range(10*num_concepts):
    best_auc = 0
    best_auc_ind = -1
    best_auc_concept = -1
    for c in range(num_concepts):
        for ind in topkinds[c]:
            # add 1 feature to test AUC
            if not condition[c][ind]:
                condition[c][ind]=True
                temp = torch.nn.Parameter(logregbottleneck.model.bottleneck.weight.clone().detach())
                logregbottleneck.model.bottleneck.weight = torch.nn.Parameter(logregbottleneck.model.bottleneck.weight.where(condition, torch.tensor(0.0).cuda()))

                # get AUC with added feature
                curr_auc = getAUC(logregbottleneck.model,X_test,y_test)
                if (curr_auc > best_auc):
                    best_auc = curr_auc
                    best_auc_ind = ind
                    best_auc_concept = c

                # remove feature
                condition[c][ind]=False
                logregbottleneck.model.bottleneck.weight = temp

    condition[best_auc_concept][best_auc_ind] = True
    best_aucs.append(best_auc)
    best_auc_inds.append(best_auc_ind)
    best_auc_concepts.append(best_auc_concept)

filename = "./models/LOS-6-600/cos-sim/top-k/vasopressor_bottleneck_r{}_c{}_topkinds.csv".format(FLAGS.split_random_state,num_concepts)

# writing to csv file
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Best AUC", "Best AUC Concept #", "Best AUC ind #"])
    # writing the data rows 
    for row in zip(best_aucs,best_auc_concepts,best_auc_inds):
        csvwriter.writerow(list(row))

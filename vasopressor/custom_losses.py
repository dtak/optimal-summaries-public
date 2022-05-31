import numpy as np
import pandas as pd
import pickle
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits

def custom_bce_l2(y_hat, y_true, network_weights, p_weight, alpha=1e-6):
    bce_term = binary_cross_entropy_with_logits(y_hat, y_true, pos_weight = p_weight)
    
    # exclude intercept weight
    l2 = torch.norm(network_weights[0])

    return bce_term + alpha * l2
    

# Modified 11/3 to use different in weights instead of L1
def custom_bce_l1(y_hat, y_true, network_weights, p_weight, alpha=5e-4):
    bce_term = binary_cross_entropy_with_logits(y_hat, y_true, pos_weight = p_weight)
    
    # Difference between layer 1 and layer 0
    l1 = torch.sum(torch.abs(network_weights[1] - network_weights[0]))

    return bce_term + alpha * l1

# Using horseshoe regularization term defined on 
# https://link.springer.com/article/10.1007/s13571-019-00217-7
def custom_bce_horseshoe(y_hat, y_true, network_weights, p_weight, alpha=1e-4, tau=10.):
    eps = 0.001
    bce_term = binary_cross_entropy_with_logits(y_hat, y_true, pos_weight = p_weight)
    frac_term = 1. / ((network_weights[1] - network_weights[0])**2 + eps)
    term_inside_log = 1 + 2 * tau**2 * frac_term
    l_hs = - torch.sum(torch.log(torch.log(term_inside_log)))

    return bce_term + alpha * l_hs

def LSTM_compound_loss(x_true, x_pred, y_true, y_pred, mask, model, p_weight, beta=10., alpha = 1e-4):
    # Custom masked MSE loss for next state prediction, plus BCE with logits for outcome prediction.
    x_loss = ((x_true - x_pred)**2)
    x_loss_masked = x_loss.masked_select(mask).mean()
    
    y_loss = binary_cross_entropy_with_logits(y_pred, y_true, pos_weight = p_weight)
    
    # Add regularization term for all of the weights of the model.
    
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg = l2_reg + torch.norm(param)

    return x_loss_masked + beta * y_loss + alpha * l2_reg

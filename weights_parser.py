import torch
import torch.nn as nn
import numpy as np

class WeightsParser(object):
    """A helper class to index into a parameter vector."""

    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += shape
        self.idxs_and_shapes[name] = (start, self.num_weights)

    def get_linear_weights(self, model, name):
        start_i, end_i = self.idxs_and_shapes[name]
        return model.linear.weight.data[0, start_i:end_i]
    
    def get_cs(self, model, name):
        start_i, end_i = self.idxs_and_shapes[name]
        return model.cutoff_times[0, start_i:end_i]
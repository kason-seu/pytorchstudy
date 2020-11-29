from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch
class LR(nn.Module):
    def __init__(self, inputs_size, output_size):
        super(LR,self).__init__()
        self.inputs_size = inputs_size
        self.output_size = output_size
        self.linear = nn.Linear(inputs_size, output_size)

    def forward(self, x):
        out = self.linear.forward(x)
        return out


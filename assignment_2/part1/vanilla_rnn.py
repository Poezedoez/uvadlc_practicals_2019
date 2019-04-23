################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.num_hidden = num_hidden
        self.seq_length = seq_length
        self.Whx = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, 0.0001, (input_dim, num_hidden))).float(), requires_grad=True)
        self.Whh = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, 0.0001, (input_dim, num_hidden))).float(), requires_grad=True)
        self.Wph = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, 0.0001, (num_hidden, num_classes))).float(), requires_grad=True)
        self.bh = torch.nn.Parameter(torch.from_numpy(np.zeros((1, num_hidden))).float(), requires_grad=True)
        self.bp = torch.nn.Parameter(torch.from_numpy(np.zeros((1, num_classes))).float(), requires_grad=True)

    def forward(self, x):   
        ht = torch.zeros((x.shape[0], self.num_hidden))
        for t in range(0, self.seq_length):
            ht = torch.tanh(x[:, t].unsqueeze(1) @ self.Whx+ self.bh.t() + ht @ self.Whh.t())
            pt = ht.t() @ self.Wph + self.bp
        return pt
        

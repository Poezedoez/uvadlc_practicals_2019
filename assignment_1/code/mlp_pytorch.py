"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
from collections import OrderedDict

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """
    modules = []
    input_size = n_inputs
    depth = 1

    ## Add hidden modules
    for output_size in n_hidden:
      modules.append(('Linear{0}'.format(depth), nn.Linear(input_size, output_size).cuda()))
      modules.append(('Relu{0}'.format(depth), nn.ReLU().cuda()))
      depth += 1
      input_size = output_size

    ##  Add output modules
    modules.append(('Linear{0}'.format(depth), nn.Linear(input_size, n_classes).cuda()))
    modules.append(('LogSoftMax', nn.LogSoftmax(dim=-1).cuda()))

    self.model = nn.Sequential(OrderedDict(modules)).cuda()


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    out = self.model(x)

    return out

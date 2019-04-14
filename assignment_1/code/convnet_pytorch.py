"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch
from collections import OrderedDict

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    super(ConvNet, self).__init__()

    self.layer1 = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(n_channels, 64, 3, padding=1).cuda()),
          ('batchnorm1', nn.BatchNorm2d(64).cuda()),
          ('relu1', nn.ReLU().cuda()),
          ('maxpool1', nn.MaxPool2d(3, stride=2, padding=1).cuda())
        ]))
    self.layer2 = nn.Sequential(OrderedDict([
          ('conv2', nn.Conv2d(64, 128, 3, padding=1).cuda()),
          ('batchnorm2', nn.BatchNorm2d(128).cuda()),
          ('relu2', nn.ReLU().cuda()),
          ('maxpool2', nn.MaxPool2d(3, stride=2, padding=1).cuda())
        ]))
    self.layer3 = nn.Sequential(OrderedDict([
          ('conv3_a', nn.Conv2d(128, 256, 3, padding=1).cuda()),
          ('batchnorm3_a', nn.BatchNorm2d(256).cuda()),
          ('relu3_a', nn.ReLU().cuda()),
          ('conv3_b', nn.Conv2d(256, 256, 3, padding=1).cuda()),
          ('batchnorm3_b', nn.BatchNorm2d(256).cuda()),
          ('relu3_b', nn.ReLU().cuda()),
          ('maxpool3', nn.MaxPool2d(3, stride=2, padding=1).cuda())
        ]))
    self.layer4 = nn.Sequential(OrderedDict([
          ('conv4_a', nn.Conv2d(256, 512, 3, padding=1).cuda()),
          ('batchnorm4_a', nn.BatchNorm2d(512).cuda()),
          ('relu4_a', nn.ReLU().cuda()),
          ('conv4_b', nn.Conv2d(512, 512, 3, padding=1).cuda()),
          ('batchnorm4_b', nn.BatchNorm2d(512).cuda()),
          ('relu4_b', nn.ReLU().cuda()),
          ('maxpool4', nn.MaxPool2d(3, stride=2, padding=1).cuda())
        ]))
    self.layer5 = nn.Sequential(OrderedDict([
          ('conv5_a', nn.Conv2d(512, 512, 3, padding=1).cuda()),
          ('batchnorm5_a', nn.BatchNorm2d(512).cuda()),
          ('relu5_a', nn.ReLU().cuda()),
          ('conv5_b', nn.Conv2d(512, 512, 3, padding=1).cuda()),
          ('batchnorm5_b', nn.BatchNorm2d(512).cuda()),
          ('relu5_b', nn.ReLU().cuda()),
          ('maxpool5', nn.MaxPool2d(3, stride=2, padding=1).cuda())
        ]))
    self.layer6 = nn.AvgPool2d(1, stride=1, padding=0).cuda()
    self.layer7 = nn.Linear(512, n_classes).cuda()


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

    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(torch.squeeze(out))

    return out

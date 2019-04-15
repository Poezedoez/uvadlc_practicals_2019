"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    self.params = {'weight': np.random.normal(0, 0.0001, (in_features, out_features)), 'bias': np.zeros(out_features)}
    self.grads = {'weight': np.zeros((in_features, out_features)), 'bias': np.zeros(out_features)}

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    out = np.dot(x, self.params['weight']) + self.params['bias']

    self.last_input = x
    self.last_output = out

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """
    self.grads['bias'] = dout
    self.grads['weight'] = np.dot(np.transpose(self.last_input), dout) # might need to transpose something here
    dx = np.dot(dout, self.params['weight'].T)
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    self.mask = (x > 0)
    out = x * self.mask

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    dx = dout * self.mask
    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    b = np.max(x)
    xi = np.exp(x - b)
    out = xi/np.sum(xi, axis=0)
    self.last_output = out

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """
    # print("dout", dout.shape)
    # print()
    # x = self.last_output.T
    # print("b", .shape)
    # print()
    # diag_3d = np.eye(b.shape[1]) * b[:,np.newaxis,:]
    # print()
    # diag_3d_sum = np.einsum('abc, ade -> de', diag_3d, diag_3d)
    # print("diag3d", diag_3d)
    # print()
    # b = self.last_output[np.newaxis, :, :]
    # outer_3d = np.einsum('aji, ejk -> jk', b, b)
    # print("in", self.last_output.shape)
    # b = self.last_output.T
    # outer_3d = b[:, :, np.newaxis] * b[:, np.newaxis, :]
    # softmax_grad_3d = diag_3d - outer_3d
    # print("reshape", dout[np.newaxis, :, :].shape)
    # sol = dx_3d * dout[np.newaxis, :, :]
    # print("dx3d", dx_3d.shape)
    # dx_sum = np.einsum('bcd, ace -> cd', dout[np.newaxis, :, :], softmax_grad_3d)
    # dx_3d_mean = np.mean(dx_3d, axis=0)
    # print("dx_3d_sum", dx_3d_sum.shape) 
    # print()
    # print("outer3d", outer_3d.shape)
    # print()
    softmax_grads = []
    x = self.last_output
    for i in range(0, x.shape[1]):
      softmax_grad = np.diagflat(x[:, i])- np.outer(x[:, i], x[:, i])
      dxi = np.dot(dout[:, i], softmax_grad)
      softmax_grads.append(dxi)

    dx = np.array(softmax_grads).T
  
    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """
    loss_total = -np.sum(np.log(x)*y, axis=1)
    out = np.mean(loss_total)

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """
    loss_total = -np.divide(y, x)
    dx = loss_total/x.shape[0]

    return dx

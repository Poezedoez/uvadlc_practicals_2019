"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """
  predicted = torch.argmax(predictions, dim=1)
  targets = torch.argmax(targets, dim=1)
  correct = (predicted == targets).float().sum()
  accuracy = correct/targets.shape[0]

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  
  # Get the datasets
  data = cifar10_utils.get_cifar10()
  n_classes = data['train'].labels.shape[1]
  mlp = MLP(data['train'].images.shape[1]*data['train'].images.shape[2]*data['train'].images.shape[3], dnn_hidden_units, n_classes)
  loss_module = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr = FLAGS.learning_rate)

  train_losses = []
  test_accuracies = []

  # Iterate over the batches
  for iteration in range(0, FLAGS.max_steps):

    optimizer.zero_grad()

    if (iteration%FLAGS.eval_freq == 0):
      print("Iteration {}...".format(iteration))
      reshaped_test = np.reshape(data['test'].images, (data['test'].images.shape[0], data['test'].images.shape[1]*data['test'].images.shape[2]*data['test'].images.shape[3]))
      test_probabilities = mlp.forward(torch.from_numpy(reshaped_test).cuda())
      acc = accuracy(test_probabilities, torch.from_numpy(data['test'].labels).cuda())
      print("Test accuracy:", acc)
      test_accuracies.append(acc)

    batch, batch_labels = data['train'].next_batch(FLAGS.batch_size)
    reshaped_batch = np.reshape(batch, (batch.shape[0], batch.shape[1]*batch.shape[2]*batch.shape[3]))
    train_probabilities = mlp.forward(torch.from_numpy(reshaped_batch).cuda())
    loss = loss_module(train_probabilities, torch.argmax(torch.from_numpy(batch_labels), dim=1).long().cuda())
    # if (iteration%FLAGS.eval_freq == 0):
    #   reshaped_train = np.reshape(data['train'].images, (data['train'].images.shape[0], data['train'].images.shape[1]*data['train'].images.shape[2]*data['train'].images.shape[3]))
    #   train_probabilities = mlp.forward(torch.from_numpy(reshaped_train).cuda())
    #   acc = accuracy(train_probabilities, torch.from_numpy(data['train'].labels).cuda())
    #   print("Train accuracy:", acc)
    #   train_losses.append(loss.item())
    loss.backward()
    optimizer.step()
  
  # Plot results
  x = range(0, len(test_accuracies)*FLAGS.eval_freq, FLAGS.eval_freq)
  fig, ax = plt.subplots()
  ax.plot(x, train_losses)
  ax.set(xlabel='batches', ylabel='loss',
        title='Loss training set after batches trained')
  ax.grid()

  fig.savefig("figures/loss_{0}_{1}_{2}_{3}.png".format(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.batch_size))
  plt.show()

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
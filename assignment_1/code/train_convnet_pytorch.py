"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  # Get everything ready
  data = cifar10_utils.get_cifar10()
  n_classes = data['train'].labels.shape[1] # 10
  n_channels = data['train'].images.shape[1] # 3
  cnn = ConvNet(n_channels, n_classes)
  loss_module = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(cnn.parameters(), lr = FLAGS.learning_rate)
  test_accuracies = []
  train_losses = []
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cnn.to(device)

  # Iterate over the batches
  for iteration in range(0, FLAGS.max_steps):

    optimizer.zero_grad()

    # Evaluate on whole test set
    if (iteration%FLAGS.eval_freq == 0):
      print("Iteration {}...".format(iteration))
      epochs = data['test'].epochs_completed
      batch_accuracies = []
      while (epochs-data['test'].epochs_completed) == 0: 
        test_batch, test_batch_labels = data['test'].next_batch(FLAGS.batch_size)
        test_probabilities = cnn.forward(torch.from_numpy(test_batch).to(device))
        acc = accuracy(test_probabilities, torch.from_numpy(test_batch_labels).to(device))
        batch_accuracies.append(acc.item())
      test_accuracy = np.mean(batch_accuracies)
      print("Test accuracy:", test_accuracy)
      test_accuracies.append(test_accuracy)

    # Train on batch
    train_batch, train_batch_labels = data['train'].next_batch(FLAGS.batch_size)
    train_probabilities = cnn.forward(torch.from_numpy(train_batch).to(device))
    loss = loss_module(train_probabilities, torch.argmax(torch.from_numpy(train_batch_labels), dim=1).long().to(device))
    if (iteration%FLAGS.eval_freq == 0):
      train_losses.append(loss.item())
    loss.backward()
    optimizer.step()

  # Plot results
  x = range(0, len(test_accuracies)*FLAGS.eval_freq, FLAGS.eval_freq)
  fig, ax = plt.subplots()
  ax.plot(x, train_losses)
  ax.set(xlabel='batches', ylabel='loss',
        title='Loss training set after batches trained')
  ax.grid()

  fig.savefig("figures/cnn_loss_{0}_{1}_{2}.png".format(FLAGS.learning_rate, FLAGS.max_steps, FLAGS.batch_size))
  # plt.show()

  x = range(0, len(test_accuracies)*FLAGS.eval_freq, FLAGS.eval_freq)
  fig, ax = plt.subplots()
  ax.plot(x, test_accuracies)
  ax.set(xlabel='batches', ylabel='accuracy',
        title='Accuracy test set after batches trained')
  ax.grid()

  fig.savefig("figures/cnn_results_{0}_{1}_{2}.png".format(FLAGS.learning_rate, FLAGS.max_steps, FLAGS.batch_size))
  # plt.show()

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
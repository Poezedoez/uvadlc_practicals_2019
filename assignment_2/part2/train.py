# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

## Use one hot encodings to prevent biases
def onehot(x, input_dim):
    new_shape = list(x.shape + (input_dim,))
    zero_hot = torch.zeros(new_shape).to(x.device)
    one_hot = zero_hot.scatter(-1, x.unsqueeze(-1), 1)

    return one_hot

## Complete a given string by generation of the model
def complete_sentence(sentence, parameters):
    sequence = parameters['dataset'].convert_to_sequence(sentence)
    sequence = torch.tensor(sequence).unsqueeze(1).to(parameters['device'])
    ht_ct = parameters['ht_ct']
    for s in range(0, sequence.shape[0]):
        next_character = sequence[s, :]
        x = onehot(next_character.view(1, -1), parameters['dataset'].vocab_size)
        _, ht_ct = parameters['model'].forward(x, ht_ct)
    
    next_character = parameters['dataset'].convert_to_sequence(' ')
    parameters.update({'ht_ct':ht_ct, 'seed':torch.tensor(next_character).unsqueeze(1).to(parameters['device'])})
    completion = generate_text(parameters)

    return sentence + completion

## Generate text up to given length given a seed and a hidden state
def generate_text(parameters):

    with torch.no_grad():
        
        text = parameters['seed'].view(-1).tolist()
        next_character = parameters['seed']
        ht_ct = parameters['ht_ct']
        for _ in range(0, parameters['sample_length']):
            x = onehot(next_character.view(1, -1), parameters['dataset'].vocab_size)
            y, ht_ct = parameters['model'].forward(x, ht_ct)
            ## Distribution dependent on temperature parameter (inverse Beta)
            distribution = torch.softmax(y[-1,:,:].squeeze()/parameters['temperature'], dim=0)
            next_character = torch.multinomial(distribution, next_character.shape[0])
            text += next_character.tolist()
    return parameters['dataset'].convert_to_string(text)


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    settings = [config.batch_size, config.seq_length, dataset.vocab_size, 
        config.lstm_num_hidden, config.lstm_num_layers, (1-config.dropout_keep_prob)]
    model = TextGenerationModel(*settings)
    model.to(device)

    print(model)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    data_iterator = iter(data_loader)
    step = 0
    epochs = 0
    results = []
    completed_sentences = []
    while True:

        # Only for time measurement of step through network
        t1 = time.time()

        # Get new batch
        try:
            batch_inputs, batch_targets = next(data_iterator)
        except StopIteration:
            epochs += 1
            print("epochs done:", epochs)
            # Reload the dataset from the start
            data_iterator = iter(data_loader)
            batch_inputs, batch_targets = next(data_iterator)
        
        # Stabilize 
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        # Forward pass
        x = onehot(torch.stack(batch_inputs), dataset.vocab_size).to(device)
        y = torch.stack(batch_targets).to(device)
        predictions, _ = model(x)
        loss = criterion(predictions.permute(0,2,1), y)

        # Backward pass
        loss.backward()
        optimizer.step()  
        accuracy = (predictions.argmax(dim=-1) == y.long()).float().mean()
        
        # Just for time measurement
        t2 = time.time()

        examples_per_second = config.batch_size/float(t2-t1)

        # print progress
        if step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss.item()
            ))

        # generate sample and complete sentence
        if step % config.sample_every == 0:
            seed = torch.randint(high=dataset.vocab_size, size=(1, 1)).to(device)
            sample_parameters = {'seed':seed, 'model':model, 'sample_length':config.seq_length, 
                'dataset':dataset, 'temperature':1, 'device':device, 'ht_ct':None}
            sample_text = generate_text(sample_parameters)
            print(sample_text)
            results.append((step, accuracy, loss.item(), sample_text))
            completed_sentence = complete_sentence(config.complete_sentence, sample_parameters)
            print(completed_sentence)
            completed_sentences.append(completed_sentence)
        
        if step == config.train_steps or epochs == config.max_epochs:
            break

        step += 1

    print('Done training.')
    torch.save(model, "models/{}_model_final.pt".format(config.txt_file))

    ## Sample with final model for different temperatures
    temperatures = [0.5, 1, 2]
    samples = []
    seeds = [torch.randint(high=dataset.vocab_size, size=(1, 1)).to(device) for _ in range(0, 5)]
    for t in temperatures:
        for seed in seeds:
            sample_parameters = {'seed':seed, 'model':model, 'sample_length':60, 
                    'dataset':dataset, 'temperature':t, 'device':device, 'ht_ct':None}
            sample_text = generate_text(sample_parameters)
            samples.append((t, seed.item(), sample_text))

    ## Save temperature samples
    with open('results/{}_temperature_samples.txt'.format(config.txt_file), 'w') as f:
        f.write("temperature, seed, sample\n")
        for temperatures, seed, sample in samples:
            f.write("%1f, %s, %s\n" %(temperatures, dataset.convert_to_string([seed]), sample))
        
    ## Save completed sentences
    with open('results/completed_sentences_{}.txt'.format(config.complete_sentence), 'w') as f:
        f.write("{}...\n".format(config.complete_sentence))
        for sentence in completed_sentences:
            f.write("%s\n" % sentence)

    ## Save results
    with open('results/{}_results.txt'.format(config.txt_file), 'w') as f:
        f.write("step, accuracy, loss, sample\n")
        for step, accuracy, loss, sample in results:
            f.write("%d, %2f, %2f, %s\n" %(step, accuracy, loss, sample))

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda", help='Device to train on')
    parser.add_argument('--max_epochs', type=int, default=1, help='Maximum amount of epochs to train')
    parser.add_argument('--complete_sentence', type=str, default="Harry, you are")
    config = parser.parse_args()

    # Train the model
    train(config)

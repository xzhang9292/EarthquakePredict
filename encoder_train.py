# NOTE: The scaffolding code for this part of the assignment
# is adapted from https://github.com/pytorch/examples.
from __future__ import print_function
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import data_loader as dler
import models.lstm_autoencoder
import models.autoencoder
import models.vanilla_autoencoder
import datetime
# Training settings
parser = argparse.ArgumentParser(description='final project')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model',
                    choices=['cnn_autoencoder','lstm_autoencoder','vanilla_autoencoder'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int,
                    help='size of convolution kernels/filters')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')
parser.add_argument('--cifar10-dir', default='data',
                    help='directory that contains cifar-10-batches-py/ '
                         '(downloaded automatically if necessary)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
'''
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
'''
data_mean = 0#4.473983593435619
data_range = 1#5444*1.5

data = dler.get_data(4096)
train_data = data['train']
test_data = data['test']
n_train_data = data['n_train']
if args.model == 'cnn_autoencoder':
    model = models.autoencoder.Autoencoder()
if args.model == 'vanilla_autoencoder':
    model = models.vanilla_autoencoder.Autoencoder()
elif args.model == 'lstm_autoencoder':
    model = models.lstm_autoencoder.Seq2seq(inputdim = 1, hidden = 100, layers = 1)
#model = torch.load('mymodel2.pt')

criterion = F.l1_loss
if args.cuda:
    model.cuda()

#############################################################################
# TODO: Initialize an optimizer from the torch.optim package using the
# appropriate hyperparameters found in args. This only requires one line.
#############################################################################
optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

def train(epoch):
    '''
    Train the model for one epoch.
    '''
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    model.train()
    # train loop
    batch_idx = 0
    for batch in dler.get_data_vector_batch(train_data, args.batch_size):
        # prepare data
        data= Variable(torch.from_numpy((batch[0]-data_mean)/data_range).float())
        if args.cuda:
            data = data.cuda()
        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.
        # This only requires a couple lines of code.
        #############################################################################
        optimizer.zero_grad()
        output = model(data)
        #print(output.size())
        loss = criterion(output, data)
        loss.backward()
        optimizer.step() 
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        if batch_idx % args.log_interval == 0:
            val_loss = evaluate('val')
            train_loss = loss.data.item()
            examples_this_epoch = batch_idx * len(data)
            epoch_progress = examples_this_epoch/n_train_data * 100#100. * batch_idx / len(train_loader)
            print(str(datetime.datetime.now()) + ' - Train Epoch: {} [{} ({:.0f}%)]\t'
                  'Train Loss: {:.10f}\tVal Loss: {:.10f}\t'.format(
                epoch, examples_this_epoch,
                epoch_progress, train_loss, val_loss))
        batch_idx += 1
        #if batch_idx == 1:
        #    break

def evaluate(split, verbose=False):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    batch_idx = 0
    for batch in dler.get_data_vector_batch(train_data, args.batch_size):
        data  = Variable(torch.from_numpy((batch[0]-data_mean)/data_range).float())
        if args.cuda:
            data = data.cuda()
        output = model(data)

        new_loss =criterion(output, data).item()
        '''
        print(data.size())
        print(output.size())
        print(data.max())
        print(output.max())
        print('new loss:')
        print(new_loss)
        '''
        loss += new_loss
        #print(output)
        # predict the argmax of the log-probabilities
        #pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += 1
        #if n_batches and (batch_i >= n_batches):
        #    break
        if(random.random()<0.001):
            print(data.data.cpu().numpy()[0])
            print(output.data.cpu().numpy()[0])

    loss /= n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}'.format(
            split, loss))
    return loss


# train the model one epoch at a time
for epoch in range(1, args.epochs + 1):
    #print(epoch)
    train(epoch)
#evaluate('val', verbose=True)

# Save the model (architecture and weights)
torch.save(model, args.model + '3.pt')
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details


# NOTE: The scaffolding code for this part of the assignment
# is adapted from https://github.com/pytorch/examples.
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import data_loader as dler
import models.reduced
import models.baseline
import models.vanilla
import datetime
import random
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
                    choices=['baseline', 'reduced','vanilla','lstm'],
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

if args.model == 'baseline':
    model = models.baseline.MyModel()
    vlength = 4096
    data_mean = 4.473983593435619
    data_range = 5444*1.5
elif args.model == 'reduced':
    model = models.reduced.MyModel()
    vlength = 100
    data_mean = 0
    data_range = 1
elif args.model == 'vanilla':
    model = models.vanilla.MyModel()
    vlength = 100
    data_mean = 0
    data_range = 1
elif args.model == 'lstm':
    model = models.lstm_model.lstmmodel(100,1,1)
    vlength = 100
    data_mean = 0
    data_range = 1
data = dler.get_data(vlength)
train_data = data['train']
test_data = data['test']
n_train_data = data['n_train']
data_stride = 1


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
    for batch in dler.get_data_batch_stride(train_data, args.batch_size, data_stride, vlength):
        # prepare data
        images, targets = Variable(torch.from_numpy((batch[0]-data_mean)/data_range).float()), Variable(torch.from_numpy(batch[1]).float())
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()
        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.
        # This only requires a couple lines of code.
        #############################################################################
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step() 
        #print(output.data.cpu().numpy())
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        #np.savetxt('test1.csv', targets.cpu().detach().numpy(),delimiter=',',fmt='%.10f')
        #np.savetxt('test2.csv', output.cpu().detach().numpy(),delimiter=',',fmt='%.10f')
        if batch_idx % args.log_interval == 0:
            val_loss = evaluate('val')
            train_loss = loss.data.item()
            #print(torch.sum(torch.abs(targets - output)).data.item())
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = examples_this_epoch/n_train_data * 100#100. * batch_idx / len(train_loader)
            print(str(datetime.datetime.now()) + ' - Train Epoch: {} [{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\t'.format(
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
    total_loss = 0
    for batch in dler.get_data_batch_stride(test_data, args.batch_size, data_stride, vlength):
        data, target = Variable(torch.from_numpy((batch[0]-data_mean)/data_range).float()), Variable(torch.from_numpy(batch[1]).float())
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        #print(output.data.cpu().numpy())
        #np.savetxt('test3.csv', target.cpu().detach().numpy(),delimiter=',',fmt='%.10f')
        #np.savetxt('test4.csv', output.cpu().detach().numpy(),delimiter=',',fmt='%.10f')
        #total_loss += torch.sum(torch.abs(target - output)).data.item()
        loss += criterion(output, target, size_average=False).data.item()
        #print(output)
        # predict the argmax of the log-probabilities
        #pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += output.size(0)
        batch_idx += 1 
        #if batch_idx==1:
        #    break
        #if(random.random()<0.001):
        #    print(target.data.cpu().numpy()[0:int(args.batch_size/10)])
        #    print(output.data.cpu().numpy()[0:int(args.batch_size/10)])
    #print(loss,n_examples)
    #print(output.size())
    #print(target.size())
    #print('total loss: %f, average loss: %f'%(total_loss, total_loss/n_examples))
    loss /= n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}'.format(
            split, loss))
    return loss


# train the model one epoch at a time
for epoch in range(1, args.epochs + 1):
    train(epoch)
#evaluate('val', verbose=True)

# Save the model (architecture and weights)
torch.save(model, args.model + '2.pt')
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details


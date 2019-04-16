import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.conv = nn.Sequential(nn.Conv1d(1,3,3),nn.ReLU(),nn.Dropout(p=0.1),nn.MaxPool1d(2,stride=2),
                                 nn.Conv1d(3,5,4),nn.ReLU(),nn.Dropout(p=0.1),nn.MaxPool1d(2,stride=2),
                                 nn.Conv1d(5,5,5),nn.ReLU(),nn.Dropout(p=0.1),nn.MaxPool1d(2, stride=2),
                                 nn.Conv1d(5,8,6),nn.ReLU(),nn.Dropout(p=0.1),nn.MaxPool1d(2, stride=2),
                                 nn.Conv1d(8,8,7),nn.ReLU(),nn.Dropout(p=0.1),nn.MaxPool1d(2, stride=2),
                                 nn.Conv1d(8,10,8),nn.ReLU(),nn.Dropout(p=0.1),nn.MaxPool1d(2, stride=2))
        self.linear = nn.Sequential(nn.Linear(22980,1000),nn.ReLU(),nn.Dropout(),
                                    nn.Linear(1000,100),nn.ReLU(),nn.Dropout(),
                                    nn.Linear(100,1),nn.ReLU())#,nn.Linear(),nn.ReLU(),
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, data):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        #print((data.view(-1,1,4096)).size())
        out = self.conv(data.view(-1,1,4096*36))
        out = out.reshape(data.size(0),-1)
        scores = self.linear(out)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores


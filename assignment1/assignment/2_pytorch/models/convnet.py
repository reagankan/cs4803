import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #Input channels = 3, output channels = 32
        in_channels = im_size[0]
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=padding)
        stride = 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=stride, padding=0)
        
        self.in_features = np.dot(hidden_dim,np.dot(im_size[1]//stride,im_size[2]//stride))
        #print(f'in_features = {self.in_features}')
       
        self.fc1 = nn.Linear(self.in_features, n_classes)
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
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
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        x = self.conv1(images)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1,self.in_features)
        scores = self.fc1(x)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Create components of a softmax classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''
        super(Softmax, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.in_features = np.dot(im_size[0],np.dot(im_size[1],im_size[2]))
        self.fc1 = nn.Linear(self.in_features, n_classes)
        #self.softmax = nn.softmax
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the classifier to
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
        # TODO: Implement the forward pass. This should take very few lines of code.
        #############################################################################
        #x = torch.flatten(images,1)
        sizes = images.size()
        #print(sizes)
        x = images.view(sizes[0],self.in_features)
        #print(f'x shape = {x.shape}')
        scores = self.fc1(x)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores


#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u test.py \
    --model mymodel.pt | tee test.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

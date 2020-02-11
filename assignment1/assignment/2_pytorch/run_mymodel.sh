#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 32 \
    --epochs 10 \
    --weight-decay 1e-5 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.001 \
    --opt adam \
    --loss softmax | tee mymodel_Adam.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

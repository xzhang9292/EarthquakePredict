#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u encoder_train.py \
    --model cnn_autoencoder \
    --kernel-size 1 \
    --hidden-dim 10 \
    --epochs 3 \
    --weight-decay 1000 \
    --momentum 0.9 \
    --batch-size 100 \
    --lr 0.000001 | tee autoencoder.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

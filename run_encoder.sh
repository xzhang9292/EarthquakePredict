#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u encoder_train.py \
    --model lstm_autoencoder \
    --kernel-size 1 \
    --hidden-dim 10 \
    --epochs 15 \
    --weight-decay 1000 \
    --momentum 0.9 \
    --batch-size 50 \
    --lr 0.000001 | tee lstm_autoencoder.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

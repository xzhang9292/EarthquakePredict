#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u train.py \
    --model mymodel \
    --kernel-size 1 \
    --hidden-dim 10 \
    --epochs 3 \
    --weight-decay 0.001 \
    --momentum 0.9 \
    --batch-size 50 \
    --lr 0.00001 | tee mymodel2tt.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

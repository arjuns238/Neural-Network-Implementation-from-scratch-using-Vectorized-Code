# Neural-Network-Implementation-from-scratch-using-Vectorized-Code

This project illustrates the forward propagation and back-propagation algorithm for neural networks and applies it to the task of hand-written digit recognition.

nnCostFunction.m starts with the cost function and gradient for the neural network. The cost function implemented with regularization is done with effecient vectorized code. The next part of nnCostFunction.m addresses backpropagation and goes through each layer to find the gradient. 

checkNNgradients.m performs gradient checking by comparing the gradients obtained in the backpropagation algorithm to the correct derivative values. 

ex4.m calls the different functions in the appropriate places to create a neural network. The built - in Ocatve library fmincg is used to train the nerual network. 


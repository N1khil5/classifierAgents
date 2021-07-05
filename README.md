# classifierAgents
 The classifier created is a simple NN with a hidden layer containing one node and an output layer 
        containing a single output node to help pacman make a decision on where to move next. 
        
 The feedforward aspect of the network transforms the input (the integers in goodmoves.txt) into an output, this
 output is then compared to the target t and if there's a difference between the output and the target, the error
 is calculated during training so the next iterations will try to minimise this error. 
        
 The backpropagation is used to correct the error using the learning rate and the weight correction term to 
 update the weights in the neural network. After this training is done for a certain amount of epochs, the 
 trained weights are used in the getAction() function and a forward pass is implemented. The output of the 
 forward pass will inform PacMan to move in a certain direction. 
        
 The Neural Network uses a sigmoid activation function at the hidden and output layer. 
        
 The network architecture and some variable naming conventions are referenced from the textbook Fundamentals of 
 Neural Networks: Architectures, Algorithms and Applications by Laurene Fausett. 
        
 v -> Array of input weights
 z -> Denotes hidden layer, variables starting with z like zW and zBias indicate the hidden layer weight and 
 hidden layer bias.
 y -> Output layer.
 alpha -> Learning rate.
 Delta -> Error correction term.
        
 Part of the NN architecture was also inspired by my undergraduate final project work where I implemented a 
 single layer perceptron to identify import features for landmarks. 

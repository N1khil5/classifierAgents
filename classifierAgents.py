# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"
        self.weights = []
        self.zweights = 0
        self.zBias = 0
    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of integers 0-3 indicating the action
        # taken in that state.
            
        # *********************************************
        #
        # Any other code you want to run on startup goes here.
        #
        # You may wish to create your classifier here.
        #
        # *********************************************
        '''
        The classifier created below is a simple NN with a hidden layer containing one node and an output layer 
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
        
        References:
        1. Fundamentals of Neural Networks by Laurene Fausett.
        2. My (Nikhil Suresh) undergraduate final year project https://github.com/N1khil5/important-landmark-features
        '''
        maxW = 1
        v = []
        alpha = 0.5
        zW = random.uniform(-maxW, maxW)
        zBias = random.uniform(0, 1)
        for j in range(150):  # Number of Epochs
            for i in range(len(self.data)):
                x = []
                inSum = 0
                corr = []
                if v == []:
                    v = [random.uniform(-maxW, maxW) for a in range(len(self.data[i]) + 1)]  # Initialising weights

                # Feedforward

                for h in range(len(self.data[i])):
                    x.append(self.data[i][h])  # Assigning each element in the array of arrays to x as an input.
                    inSum += np.dot(x[h], v[h])  # Multiplying the weight and input node and adding the result here.
                inSum += v[-1]  # Bias added

                z = (1.0 / (1.0 + np.exp(-inSum)))  # Activation function on the hidden layer node.

                # Multiplying weight of the hidden layer with the hidden layer node and adding the bias
                zSum = z * zW + zBias

                y = (3.0 / (1.0 + np.exp(-zSum)))  # Activation on output node.

                # Backpropagation

                t = self.target[i]

                delta = (t - y) * (0.5 * (1.0 + y) * (1.0 - y))  # Error correction term for hidden layer

                zCorr = alpha * delta * z  # Hidden layer correction term
                zBiasCorr = alpha * delta  # Hidden layer bias correction term

                zW = zW + zCorr  # Updating the weight for the hidden layer node
                zBias = zBias + zBiasCorr  # Updating the weight for the hidden layer bias

                deltaInput = delta * zCorr  # Calculating error for the input layer
                inputCorr = deltaInput * (0.5 * (1.0 + zSum) * (1.0 - zSum))  # Error information term for input layer

                for h in range(len(x)):
                    if len(x) == len(corr):  # If corr is already created
                        corr[h] += (alpha * inputCorr * x[h])  # Update the correction term for each input weight.
                    else:
                        # Appending weight corrections for every input term if corr is empty
                        corr.append(alpha * inputCorr * x[h])

                bCorr = alpha * inputCorr  # Calculating the bias correction term for the input layer

                for h in range(len(x)):
                    v[h] = v[h] + corr[h]  # Updating input weights
                v[-1] = v[-1] + bCorr  # Updating bias weights

        self.weights = v  # Weights from input to hidden layer which will be used for the forward pass in getAction().
        self.zweights = zW  # Weights from the hidden layer to the output node.
        self.zBias = zBias  # Bias for the hidden layer used for forward pass in getAction().

    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        
        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):
        # How we access the features.
        features = api.getFeatureVector(state)
        inSum = 0  # Initialising the sum of inputs multiplied by the weights.
        for i in range(len(features)):
            inSum += np.dot(features[i], self.weights[i])  # Adding the sum of inputs multiplied by weights.
        inSum += self.weights[-1]  # Adding the bias from the input layer to the sum of inputs * weights.
        z = (1.0 / (1.0 + np.exp(-inSum)))  # Sigmoid activation function on the hidden layer.

        zSum = z * self.zweights + self.zBias  # Sum of weight * hidden layer node + the bias of the hidden layer.

        '''
        Sigmoid activation function on the output node. The function normally has 1 in the numerator which has a range 
        of (0,1), this activation function is multiplied by 3 to get a range of (0,3). Since there are 4 possible moves
        for pacman to make, these directions will correspond to the ones defined in the convertNumberToMove() function.
        The output y is rounded to and computed in the convertNumberToMove() function and this output is returned in
        api.makeMove to move pacman in that direction.
        '''
        y = (3.0 / (1.0 + np.exp(-zSum)))

        direction = self.convertNumberToMove(round(y))


        # *****************************************************
        #
        # Here you should insert code to call the classifier to
        # decide what to do based on features and use it to decide
        # what action to take.
        #
        # *******************************************************

        # Get the actions we can try.
        legal = api.legalActions(state)

        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        return api.makeMove(direction, legal)


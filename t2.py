import numpy as np
import matplotlib.pyplot as plt
from helpers import tokenize, sigmoid, softmax, sigmoid_d
import scipy.special
import math

def codebook_transform(id, codebook):
    arr = [0] * len(codebook)
    arr[int(id)] = 1
    return arr

def net(gray_img, rgb, rgb_book, rgb_book_reverse, descent='F'):
    features = np.array([image.flatten() for image in gray_img])
    labels = np.array([i[1][1] for i in rgb])
    labels_vector = np.zeros((len(gray_img), 5)) # training labels, each in form [0. 0. 0. 1. 0.]
    for i in range(len(gray_img)):
        labels_vector[i, int(labels[i])] = 1

    hidden_nodes = 9
    output_labels = 5
    learning_rate = 0.0011
    hidden_weights = np.random.rand(features.shape[1],hidden_nodes)
    hidden_biases = np.random.randn(hidden_nodes)
    out_weights = np.random.rand(hidden_nodes,output_labels)
    out_biases = np.random.randn(output_labels)
    
    runs = 25000
    history = []

    for run in range(runs):
        #Feed-forward
        output_hidden = np.dot(features, hidden_weights) + hidden_biases #run layer 1
        output_hidden_actived = sigmoid(output_hidden)
        output_outer = np.dot(output_hidden_actived, out_weights) + out_biases #run layer 2 (outer)
        resultant = softmax(output_outer)
        
        #Back Propagate
        differential = resultant - labels_vector
        output_hidden_actived_difference = np.dot(output_hidden_actived.T, differential)
        d_hidden = np.dot(differential , out_weights.T) * sigmoid_d(output_hidden)

        if (descent == 'F'): #full
            #update
            hidden_weights -= learning_rate * np.dot(features.T, d_hidden)
            hidden_biases -= learning_rate * d_hidden.sum(axis=0)
            out_weights -= learning_rate * output_hidden_actived_difference
            out_biases -= learning_rate * differential.sum(axis=0)
        elif (descent == 'S'): #stochastic    
            x = (d_hidden[np.random.randint(0, len(d_hidden - 1))])
            x = ([x for _ in range(len(d_hidden))])
            #update

            y = output_hidden_actived_difference[np.random.randint(0, len(output_hidden_actived_difference) - 1)]
            y = ([y for _ in range(len(output_hidden_actived_difference))])
            y = np.array(y)

            hidden_weights -= learning_rate * np.dot(features.T, x)
            hidden_biases -= learning_rate * x.sum(axis=0)
            out_weights -= learning_rate * y
            out_biases -= learning_rate * y.sum(axis=0)

        if run % 100 == 0:
            loss = np.sum(-labels_vector * np.log(resultant))
            history.append(loss)
            print("Loss:", loss)         
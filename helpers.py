import numpy as np
import scipy.special

def tokenize(array):
    out = 0
    for element in array:
        out += element
        out *= 100
    return out

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    return scipy.special.expit(A) / scipy.special.expit(A).sum(axis=1, keepdims=True)
    
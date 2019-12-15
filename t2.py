import numpy as np
import matplotlib.pyplot as plt
from helpers import tokenize, sigmoid, softmax, sigmoid_d
import scipy.special
import math
import cv2

def codebook_transform(id, codebook):
    arr = [0] * len(codebook)
    arr[int(id)] = 1
    return arr

def net(gray_img, rgb, img_shape, rgb_book, rgb_book_reverse, descent='F'):

    output_labels = 5

    gray_img_full = gray_img[:]
    rgb_full = rgb[:]

    gray_img_test = gray_img[7 * len(gray_img) // 8 : len(gray_img)]
    rgb_test = rgb[7 * len(rgb) // 8 : len(rgb)]
    gray_img = gray_img[0:7 * len(gray_img) // 8]
    rgb = rgb[0:7 * len(rgb) // 8]
    features = np.array([image.flatten() for image in gray_img])
    labels = np.array([i[1][1] for i in rgb])
    labels_vector = np.zeros((len(gray_img), output_labels)) # training labels, each in form [0. 0. 0. 1. 0.]
    for i in range(len(gray_img)):
        labels_vector[i, int(labels[i])] = 1

    hidden_nodes = 30
    learning_rate = 0.0011
    hidden_weights = np.random.rand(features.shape[1],hidden_nodes)
    hidden_biases = np.random.randn(hidden_nodes)
    out_weights = np.random.rand(hidden_nodes,output_labels)
    out_biases = np.random.randn(output_labels)

    runs = 10000
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
            hidden_biases -= learning_rate * x[0].sum(axis=0)
            out_weights -= learning_rate * y
            out_biases -= learning_rate * y[0].sum(axis=0)

        divd = 100
        if run % divd == 0:
            loss = np.sum(-labels_vector * np.log(resultant))
            history.append(loss)
            print(f'Loss: (heat #{int(run/divd)}/{int(runs/divd)})', loss,flush=True)

    #testing data
    features = np.array([image.flatten() for image in gray_img_test])
    labels = np.array([i[1][1] for i in rgb_test])
    labels_vector = np.zeros((len(gray_img_test), output_labels)) # training labels, each in form [0. 0. 0. 1. 0.]
    for i in range(len(gray_img_test)):
        labels_vector[i, int(labels[i])] = 1

    output_hidden = np.dot(features, hidden_weights) + hidden_biases #run layer 1
    output_hidden_actived = sigmoid(output_hidden)
    output_outer = np.dot(output_hidden_actived, out_weights) + out_biases #run layer 2 (outer)
    resultant = softmax(output_outer)

    loss = np.sum(-labels_vector * np.log(resultant))
    print("Testing Loss:", loss)

    #final product

    features = np.array([image.flatten() for image in gray_img_full])
    labels = np.array([i[1][1] for i in rgb_full])
    labels_vector = np.zeros((len(gray_img_full), output_labels)) # training labels, each in form [0. 0. 0. 1. 0.]
    for i in range(len(gray_img_full)):
        labels_vector[i, int(labels[i])] = 1

    output_hidden = np.dot(features, hidden_weights) + hidden_biases #run layer 1
    output_hidden_actived = sigmoid(output_hidden)
    output_outer = np.dot(output_hidden_actived, out_weights) + out_biases #run layer 2 (outer)
    resultant = softmax(output_outer)

    loss = np.sum(-labels_vector * np.log(resultant))
    print("Final Loss:", loss)

    img = np.zeros( img_shape, dtype=int )
    for i in range(1,img_shape[0]-2):
        for j in range(1,img_shape[1]-2):
            index = (i-1) * (img_shape[1] - 2) + j - 1
            img[i][j] = rgb_book[np.nonzero(output_outer[index] == (max(output_outer[index])))[0][0]]

    plt.figure(1)
    plt.imshow(img)
    print(img)
    cv2.imwrite('result.png',img)

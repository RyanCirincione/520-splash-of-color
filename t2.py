import numpy as np
import matplotlib.pyplot as plt
from helpers import tokenize
import scipy.special
import math

def codebook_transform(id, codebook):
    arr = [0] * len(codebook)
    arr[int(id)] = 1
    return arr

def net(gray_img, rgb, rgb_book, rgb_book_reverse):
    print("we in here!")
    #np.random.seed(42)
    
    print("g", gray_img[0])

    feature_set = np.array([image.flatten() for image in gray_img])
    print(feature_set[-1])
    #feature_set = np.empty((0,9))
  #  for i, image in enumerate(gray_img):
   #     feature_set = np.vstack([feature_set, image.flatten()]) #middle of rgb
        #print(feature_set[-1])
    #print (feature_set)
    #cat_images = np.random.randn(700, 2) + np.array([0, -3])
    #mouse_images = np.random.randn(700, 2) + np.array([3, 3])
    #dog_images = np.random.randn(700, 2) + np.array([-3, 3])

    #feature_set = np.vstack([cat_images, mouse_images, dog_images])

    #labels = np.array([0]*700 + [1]*700 + [2]*700)
    print('labels')
    labels = np.array([i[1][1] for i in rgb])
    print(labels)

    one_hot_labels = np.zeros((len(gray_img), 5)) # "ideal" answers, each in form [0. 0. 0. 1. 0.]
    for i in range(len(gray_img)):
        one_hot_labels[i, int(labels[i])] = 1

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(x):
        return sigmoid(x) *(1-sigmoid (x))

    def softmax(A):
        expA = scipy.special.expit(A)
        return expA / expA.sum(axis=1, keepdims=True)

    instances = feature_set.shape[0]
    attributes = feature_set.shape[1]
    hidden_nodes = 9
    output_labels = 5

    wh = np.random.rand(attributes,hidden_nodes)
    bh = np.random.randn(hidden_nodes)

    wo = np.random.rand(hidden_nodes,output_labels)
    bo = np.random.randn(output_labels)
    lr = 10e-4

    error_cost = []

    for epoch in range(50000):
    ############# feedforward

        # Phase 1
        zh = np.dot(feature_set, wh) + bh
        ah = sigmoid(zh)

        # Phase 2
        zo = np.dot(ah, wo) + bo
        ao = softmax(zo)

    ########## Back Propagation

    ########## Phase 1

        dcost_dzo = ao - one_hot_labels
        dzo_dwo = ah

        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

        dcost_bo = dcost_dzo

    ########## Phases 2

        dzo_dah = wo
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        dah_dzh = sigmoid_der(zh)
        dzh_dwh = feature_set
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        dcost_bh = dcost_dah * dah_dzh

        # Update Weights ================

        wh -= lr * dcost_wh
        bh -= lr * dcost_bh.sum(axis=0)

        wo -= lr * dcost_wo
        bo -= lr * dcost_bo.sum(axis=0)

        if epoch % 1000 == 0:
            loss = np.sum(-one_hot_labels * np.log(ao))
            print('Loss function value: ', loss)
            error_cost.append(loss)
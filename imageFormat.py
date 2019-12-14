import numpy as np
from skimage import data
import matplotlib.pylab as plt
from skimage.color import rgb2gray
import skimage.io
import cv2#we only use this to read the input image
from t2 import net
from helpers import tokenize

def traverse(array):
    output = list()
    bud_size = 3
    for i in range(len(array) - bud_size + 1):
        for j in range(len(array[0]) - bud_size + 1):
            output.append(array[i:i+bud_size,j:j+bud_size])  
    return output

cam = cv2.imread('input/coffee_5.png')#data.coffee()
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
#cam = data.coffee()

colorList = np.unique(cam.reshape(-1, cam.shape[2]), axis=0)
print("colors:",len(colorList))

color_codebook = dict()
color_reverse_codebook = dict()
for i, color in enumerate(colorList):
    color_codebook[i] = color
    color_reverse_codebook[tokenize(color)] = i

reduced_cam = np.zeros((cam.shape[0], cam.shape[1]))
for i in range(cam.shape[0]):
    for j in range(cam.shape[1]):
        reduced_cam[i][j] = color_reverse_codebook[tokenize(cam[i][j])]

#print(reduced_cam)

a,b,c = cam.shape
offset = 1
newCam = np.zeros((a+2*offset,b+2*offset,c))

for i in range(cam.shape[0]):
    for j in range(cam.shape[1]):
       if abs(cam.shape[0] - i) <= offset or abs(cam.shape[1] - j) <= offset or i < offset or j < offset:
           cam[i][j] = np.array([255, 255, 255])

cam_gray = rgb2gray(cam)

correct_images = traverse(reduced_cam)

input_data = traverse(cam_gray)

print (correct_images[-1][0][0])
print (input_data[-1][0][0])

#correct_images  = correct_images[0:len(correct_images)//10]
#input_data      = input_data[0:len(input_data)//10]
net(input_data, correct_images, cam.shape, color_codebook, color_reverse_codebook)

plt.gray()
plt.imshow(cam_gray)
plt.show()
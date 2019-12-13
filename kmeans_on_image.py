print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from skimage import data
import cv2

n_colors = 16

china = cv2.imread('input/waterfall1.png')
china = cv2.cvtColor(china, cv2.COLOR_BGR2RGB)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

# print(np.mean(np.array(
# [[[1, 10], [2, 20], [3, 30]], [[4, 40], [5, 50], [6, 60]], [[7, 70], [8, 80], [9, 90]]]
# ), axis=1), flush = True)

def distance(d1, d2):
    return np.linalg.norm(d1 - d2)

# Input:
#  data - 2D array, rows of data points
#  num_clusters - number of kmean clusters
# Output:
#  labels - 1D array, parallel to data, but contains indexes in centers that best represent the parallel data point
#  centers - The centers of the kmean clusters
def kmeans(data, num_clusters):
    data_copy = np.copy(data)
    np.random.shuffle(data_copy)
    sample = data_copy[:num_clusters * 200]
    init_sample = np.copy(sample)
    init_sample.shape = (num_clusters, 200, sample.shape[1]) # Segment the data points in groups relative to their centers
    centers = np.mean(init_sample, axis=1)
    labels = np.zeros(sample.shape[0], dtype=int)

    while True:
        new_centers = np.zeros(centers.shape)
        group_counts = np.zeros(num_clusters)
        # Have every point find its nearest center, then have it contribute to the new centers
        for i in range(sample.shape[0]):
            # if i % 10 == 0:
            #     print(1.0 * i / sample.shape[0])
            for j in range(num_clusters):
                if distance(centers[labels[i]], sample[i]) > distance(centers[j], sample[i]):
                    labels[i] = j

            new_centers[labels[i]] = np.multiply(new_centers[labels[i]], group_counts[labels[i]])
            new_centers[labels[i]] = np.add(new_centers[labels[i]], sample[i])
            group_counts[labels[i]] += 1
            new_centers[labels[i]] = np.multiply(new_centers[labels[i]], 1 / group_counts[labels[i]])

        if np.array_equal(centers, new_centers):
            return centers, labels
        else:
            centers = new_centers

def predict(data, centers):
    labels = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        for j in range(centers.shape[0]):
            if distance(centers[labels[i]], data[i]) > distance(centers[j], data[i]):
                labels[i] = j

    return labels

# Load the Summer Palace photo
#china = data.coffee()
china = cv2.imread('input/waterfall1.png')
china = cv2.cvtColor(china, cv2.COLOR_BGR2RGB)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
# kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
cluster_centers, labels = kmeans(image_array, n_colors)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = predict(image_array, cluster_centers)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:n_colors]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(cluster_centers, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()

plt.imsave(f'input/waterfall1_{n_colors}.png', recreate_image(cluster_centers, labels, w, h))

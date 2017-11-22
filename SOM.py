# Image Compression using SOM

import numpy as np
from scipy import misc, spatial
from math import log, exp, pow
from sklearn.metrics import mean_squared_error
import cv2
from math import sqrt
from scipy.cluster.vq import vq
import scipy.misc
import sys


def mse(image_a, image_b):
    # calculate mean square error between two images
    err = np.sum((image_a.astype(float) - image_b.astype(float)) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err


class SOM(object):

    def __init__(self, rows, columns, dimensions, epochs, number_of_input_vectors, alpha, sigma):

        self.rows = rows
        self.columns = columns
        self.dimensions = dimensions
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.number_of_input_vectors = number_of_input_vectors
        self.number_of_iterations = self.epochs * self.number_of_input_vectors

        self.weight_vectors = np.random.uniform(0, 255, (self.rows * self.columns, self.dimensions))

    def get_bmu_location(self, input_vector, weights):

        tree = spatial.KDTree(weights)
        bmu_index = tree.query(input_vector)[1]
        return np.array([int(bmu_index/self.columns), bmu_index % self.columns])

    def update_weights(self, iter_no, bmu_location, input_data):

        learning_rate_op = 1 - (iter_no/float(self.number_of_iterations))
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        distance_from_bmu = []
        for x in range(self.rows):
            for y in range(self.columns):
                distance_from_bmu = np.append(distance_from_bmu, np.linalg.norm(bmu_location - np.array([x, y])))

        neighbourhood_function = [exp(-0.5 * pow(val, 2) / float(pow(sigma_op, 2))) for val in distance_from_bmu]

        final_learning_rate = [alpha_op * val for val in neighbourhood_function]

        for l in range(self.rows * self.columns):
            weight_delta = [val*final_learning_rate[l] for val in (input_data - self.weight_vectors[l])]
            updated_weight = self.weight_vectors[l] + np.array(weight_delta)
            self.weight_vectors[l] = updated_weight

    def train(self, input_data):

        iter_no = 0
        for epoch_number in range(self.epochs):
            for index, input_vector in enumerate(input_data):
                bmu_location = self.get_bmu_location(input_vector, self.weight_vectors)
                self.update_weights(iter_no, bmu_location, input_vector)
                iter_no += 1
        return self.weight_vectors

# source image
image_location = sys.argv[1]
image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
image_height = len(image)
image_width = len(image[0])

# dimension of the vector
block_width = int(sys.argv[3])
block_height = int(sys.argv[4])
vector_dimension = block_width*block_height

bits_per_codevector = int(sys.argv[2])
codebook_size = pow(2, bits_per_codevector)

epochs = int(sys.argv[5])

initial_learning_rate = float(sys.argv[6])

# dividing the image into 4*4 blocks of pixels
image_vectors = []
for i in range(0, image_height, block_height):
    for j in range(0, image_width, block_width):
        image_vectors.append(np.reshape(image[i:i+block_width, j:j+block_height], vector_dimension))
image_vectors = np.asarray(image_vectors).astype(float)
number_of_image_vectors = image_vectors.shape[0]

# properties of the SOM grid
som_rows = int(pow(2, int((log(codebook_size, 2))/2)))
som_columns = int(codebook_size/som_rows)

som = SOM(som_rows, som_columns, vector_dimension, epochs, number_of_image_vectors,
          initial_learning_rate, max(som_rows, som_columns)/2)
reconstruction_values = som.train(image_vectors)

image_vector_indices, distance = vq(image_vectors, reconstruction_values)

image_after_compression = np.zeros([image_width, image_height], dtype="uint8")
for index, image_vector in enumerate(image_vectors):
    start_row = int(index / (image_width/block_width)) * block_height
    end_row = start_row + block_height
    start_column = (index*block_width) % image_width
    end_column = start_column + block_width
    image_after_compression[start_row:end_row, start_column:end_column] = \
        np.reshape(reconstruction_values[image_vector_indices[index]],
                   (block_width, block_height))

output_image_name = "CB_size=" + str(codebook_size) + ".png"
scipy.misc.imsave(output_image_name, image_after_compression)

print "Mean Square Error = ", mse(image, image_after_compression)

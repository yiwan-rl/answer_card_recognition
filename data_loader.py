########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import os
import cv2

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/card/"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Height and width of each image.
img_height = 64
img_width = 36

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Number of problems in one card
num_problems = 15

# Choices
choices = ['A', 'B', 'C', 'D']

# Number of problems in one problem
num_choices = 4

# Dimensionality of label
dim_label = num_problems * num_choices

########################################################################
# Various constants used to allocate arrays of the correct size.

# Total number of images in the training-set.
_num_images_train = 1000

# Total number of images in the testing-set.
_num_images_test = 100
########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_img_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.

    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, 'img', filename)


def _get_label_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.

    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, 'label', filename)


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 3-dim array with shape: [height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0
    return cv2.resize(raw_float, (img_width, img_height))


def _one_hot_encoded(classes):
    return classes.reshape(classes.shape[0], -1)


def _get_label(file_path):
    """
    get class label from file stored in file_path
    the class label is a 2-dimensional array, e.g.
    1 0 0 1
    1 0 0 0
    0 0 0 0
    1 1 1 1
    ...
    which can be easily interpreted to readable answers,
    A D
    A

    A B C D
    ...
    and can be easily encoded in vector
    1 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 ...
    """
    label = np.zeros(shape=[num_problems, num_choices])
    with open(file_path) as f:
        for l in f.readlines():
            tmp = l.strip().split()
            if len(tmp) != 2:
                continue
            idx = int(tmp[0])
            for i in range(num_choices):
                if choices[i] in tmp[1]:
                    label[idx - 1, i] = 1
    return label


def _load_data(img_name, label_name):
    """
    Load an img and corresponding label file
    and return the converted image and the class-number for that image.
    """
    img = cv2.imread(_get_img_file_path(img_name))
    label = _get_label(_get_label_file_path(label_name))

    return _convert_images(img), label


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_height, img_width, num_channels], dtype=float)
    classes = np.zeros(shape=[_num_images_train, num_problems, num_choices], dtype=int)

    # For each image-label pair.
    for i in range(_num_images_train):
        # Load the images and class-numbers from the data-file.
        image, cls = _load_data(img_name=str(i + 1).zfill(4) + '.jpg', label_name=str(i + 1).zfill(4) + '.txt')

        # Store the images into the array.
        images[i, :] = image

        # Store the class-numbers into the array.
        classes[i, :] = cls

    return images, classes, _one_hot_encoded(classes)


def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_test, img_height, img_width, num_channels], dtype=float)
    classes = np.zeros(shape=[_num_images_test, num_problems, num_choices], dtype=int)
    for i in range(_num_images_test):
        curr_idx = _num_images_train + i + 1
        image, cls = _load_data(img_name=str(curr_idx).zfill(4) + '.jpg', label_name=str(curr_idx).zfill(4) + '.txt')
        # Store the images into the array.
        images[i, :] = image

        # Store the class-numbers into the array.
        classes[i, :] = cls

    return images, classes, _one_hot_encoded(classes)

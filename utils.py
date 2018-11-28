import numpy as np
import imageio
from keras.utils import Sequence
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from skimage.transform import resize


def create_VEDAI(PATH_TO_VEHICLES_FOLDER):
    # Takes in the full path to the unzipped "VEHICULES" folder
    # Returns RGB and Infrared images in (images, x, y, channels) format
    NUM_FILES = 2536
    MAX_INDEX = 1272
    X_PIXELS = 1024
    Y_PIXELS = 1024

    onlyfiles = [f for f in listdir(PATH_TO_VEHICLES_FOLDER) if isfile(
        join(PATH_TO_VEHICLES_FOLDER, f)) and "png" in f]
    assert len(onlyfiles) == NUM_FILES, "Not the full VEDAI 1024 Dataset"
    rgb = np.zeros((MAX_INDEX, X_PIXELS, Y_PIXELS, 3))
    infra = np.zeros((MAX_INDEX, X_PIXELS, Y_PIXELS, 1))
    indices = [format(n, '08') for n in range(MAX_INDEX)]
    for index in indices:
        if str(index) in onlyfiles:
            pair = [file for file in onlyfiles if str(index) in file]
            for file in pair:
                im = imageio.imread(file)
                if "co" in file:
                    rgb[int(index), :, :, :] = np.reshape(
                        im, (tuple([1]) + im.shape))
                elif "ir" in file:
                    infra[int(index), :, :, :] = np.reshape(
                        im, (tuple([1]) + im.shape + tuple([1])))
        else:
            print("The following image is missing!" + index)

    return rgb, infra


def save_VEDAI(rgb, infra):
    # Takes in arrays of rgb and infrared images
    # Saves them to disk, no return value
    np.save("vedai_rgb_all.npy", rgb)
    np.save("vedai_infra_all.npy", infra)


def load_VEDAI():
    # No parameters, expected to run in directory with VEDAI.npy files
    # Returns two arrays with rgb and infrared images respectively
    rgb = np.load("vedai_rgb_all.npy")
    infra = np.load("vedai_infra_all.npy")
    return rgb, infra


def data_explore(data):
    print("Shape of the data is" + str(data.shape))
    print("Dtype of the data is" + str(data.dtype))


def combine_rgb_infra(rgb, infra):
    # Concatenates the two modalities along the channels axis
    four_channel = np.concatenate(rgb, infra, axis=-1)
    return four_channel


def overlapping_patches(images, patch_size=(64, 64), padding="VALID"):
    sess = tf.Session()

    num_images, size_x, size_y, channels = images.shape
    ims = tf.convert_to_tensor(images)
    patch_x, patch_y = patch_size
    patches = tf.extract_image_patches(ims, [1, patch_x, patch_y, 1], [
        1, patch_x, patch_y, 1], [1, 1, 1, 1], padding=padding)
    patches_shape = tf.shape(patches)
    with sess.as_default():
        np = tf.reshape(patches, [tf.reduce_prod(patches_shape[0:3]),
                                  patch_x, patch_y, channels]).eval()
        return np


def non_overlapping_patches(image, patch_size=(64, 64)):
    size_x, size_y, channels = image.shape
    patch_x, patch_y = patch_size
    im_pad = np.pad(image, ((0, size_x % patch_x),
                            (0, size_y % patch_y), (0, 0)), mode="constant")
    num_patches = (size_x // patch_x + 1) * (size_y // patch_y + 1)
    patches = np.zeros(num_patches, patch_x, patch_y, channels)
    counter = 0
    for i in range((size_x // patch_x) + 1):
        for j in range((size_y // patch_y) + 1):
            x_s = i * patch_x
            y_s = j * patch_y
            patches[counter, :, :, :] = im_pad[x_s:x_s + patch_x - 1,
                                               y_s:y_s + patch_y - 1, :]
            counter += 1
    return patches


def downsample_image(image, factor=4):
    # Downsamples numpy array image by factor
    # Returns the image and the downsampled copy in a tuple
    h, w = image.size
    h = h // factor
    w = w // factor
    return image, resize(image, (h, w))


def reconstruct_patches(patches, image_size):
    # TODO: Create a function which reconstructs an image
    # when given patches created by non_overlapping_patches
    # Discards predictions for zero border
    pass


class VEDAISequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, augmentations):
        # TODO: Initialize keras augmentor, save self variables
        pass

    def __len__(self):
        # TODO: Return number of batches given data and augmentations
        pass

    def __getitem__(self, idx):
        # TODO: Return a batch of correct size from augmentor
        pass

    def on_epoch_end(self):
        # TODO: Reseed the augmentor with new augmentations
        pass

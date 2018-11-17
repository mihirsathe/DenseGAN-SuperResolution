import numpy as np
import imageio
from keras.utils import Sequence
from os import listdir
from os.path import isfile, join


def create_VEDAI(PATH_TO_VEHICLES_FOLDER):
    # Takes in the full path to the unzipped "VEHICULES" folder
    # Returns RGB and Infrared images in (images, x, y, channels) format
    NUM_FILES = 2536
    X_PIXELS = 1024
    Y_PIXELS = 1024

    onlyfiles = [f for f in listdir(PATH_TO_VEHICLES_FOLDER) if isfile(
        join(PATH_TO_VEHICLES_FOLDER, f)) and "png" in f]
    assert len(onlyfiles) == NUM_FILES, "Not the full VEDAI 1024 Dataset"
    rgb = np.zeros((NUM_FILES, X_PIXELS, Y_PIXELS, 3))
    infra = np.zeros((NUM_FILES, X_PIXELS, Y_PIXELS, 1))
    indices = [format(n, '08') for n in range(NUM_FILES)]
    for index in indices:
        pair = [file for file in onlyfiles if str(index) in file]
        for file in pair:
            im = imageio.imread(file)
            if "co" in file:
                rgb[int(index), :, :, :] = np.reshape(im,
                                                      (tuple([1]) + im.shape))
            elif "ir" in file:
                infra[int(index), :, :, :] = np.reshape(
                    im, (tuple([1]) + im.shape + tuple([1])))
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


def non_overlapping_patches(images, patch_size=(64, 64), padding="reflect"):
    # TODO: Create function that returns patches given images and patch size
    pass


def reconstruct_patches(patches, image_size, padding="reflect"):
    # TODO: Create a function which reconstructs an image
    # when given patches created by non_overlapping_patches
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

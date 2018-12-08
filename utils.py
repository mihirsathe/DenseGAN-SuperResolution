import numpy as np
import imageio
from keras.utils import Sequence
from os import listdir
from os.path import isfile, join
from skimage.transform import downscale_local_mean


def create_VEDAI(PATH_TO_VEHICLES_FOLDER):
    # Takes in the full path to the unzipped "VEHICULES" folder
    # Returns list of filenames of both rgb and infrared images
    NUM_FILES = 2536
    MAX_INDEX = 1272
    onlyfiles = [f for f in listdir(PATH_TO_VEHICLES_FOLDER) if isfile(
        join(PATH_TO_VEHICLES_FOLDER, f)) and "png" in f]
    assert len(onlyfiles) == NUM_FILES, "Not the full VEDAI 1024 Dataset"
    rgb = []
    infra = []
    indices = [format(n, '08') for n in range(MAX_INDEX)]
    missing_offset = 0
    for index in indices:
        pair = [file for file in onlyfiles if str(index) in file]
        if pair:
            for file in pair:
                if "co" in file:
                    rgb[int(index) - missing_offset] = file
                elif "ir" in file:
                    infra[int(index) - missing_offset] = file
        else:
            print("The following image is missing!: " + index)
            missing_offset += 1
    assert len(rgb) == len(infra), "Not every file has its pair!"
    assert len(rgb) == NUM_FILES / 2, "Didn't save every image."
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


def non_overlapping_patches(image, patch_size=(64, 64)):
    size_x, size_y, channels = image.shape
    patch_x, patch_y = patch_size
    im_pad = np.pad(image, ((0, patch_x - size_x % patch_x),
                            (0, patch_y - size_y % patch_y), (0, 0)),
                    mode="constant")
    if size_x % patch_x == 0 and size_y % patch_y == 0:
        im_pad = image
    pad_x, pad_y, channels = im_pad.shape
    print(im_pad.shape)
    num_patches = (pad_x // patch_x) * (pad_y // patch_y)
    patches = np.zeros((num_patches, patch_x, patch_y, channels))
    counter = 0
    for i in range((pad_x // patch_x)):
        for j in range((pad_y // patch_y)):
            x_s = i * patch_x
            y_s = j * patch_y
            patches[counter, :, :, :] = im_pad[x_s:x_s + patch_x,
                                               y_s:y_s + patch_y, :]
            counter += 1
    return patches


def downsample_image(image, block=[2, 2, 1]):
    # Downsamples numpy array image by factor
    # Returns the image and the downsampled copy in a tuple
    return image, downscale_local_mean(image, block)


def reconstruct_patches(patches, image_size):
    # TODO: Create a function which reconstructs an image
    # when given patches created by non_overlapping_patches
    # Discards predictions for zero border
    pass


class VEDAISequence(Sequence):

    def __init__(self, rgb, infra, ims_per_batch):
        # TODO: Initialize keras augmentor
        self.r, self.i = rgb, infra
        self.batch_size = ims_per_batch

    def __len__(self):
        return int(np.ceil(len(self.r) / float(self.batch_size)))

    def __getitem__(self, idx):
        batchsz = 256 * self.batch_size
        channels = 4
        patch_x, patch_y = 64, 64
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        high_res = np.zeros((batchsz, patch_x, patch_y, channels))
        low_res = np.zeros((batchsz, patch_x // 2, patch_y // 2, channels))
        im_num = 0
        for ind in range(start, end):
            st, stp = im_num * 256, (im_num + 1) * 256
            im_num += 1
            rgb = imageio.imread(self.r[ind])
            infra = imageio.imread(self.i[ind])
            high_res[st:stp, :, :, :] = non_overlapping_patches(
                combine_rgb_infra(rgb, infra))
        for hr in range(batchsz):
            low_res[hr, :, :, :] = downsample_image(high_res)
        return low_res, high_res

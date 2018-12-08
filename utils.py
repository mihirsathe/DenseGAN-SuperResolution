import numpy as np
import imageio
from keras.utils import Sequence
from os import listdir
from os.path import isfile, join
from skimage.transform import downscale_local_mean
import random

def normalize(image):
  return image/255.0    

def read_VEDAI(subset, PATH_TO_VEHICLES_FOLDER):
    # Takes in the full path to the unzipped "VEHICULES" folder
    # Returns mapping dict, RGB and Infrared images in 
    # (images, x, y, channels) format,
    # saves a txt file with mapping of rgb/infra idx to filename
    # NUM_FILES = 2536
    # MAX_INDEX = 1272
    X_PIXELS = 1024
    Y_PIXELS = 1024
    PATH_TO_VEHICLES_FOLDER = PATH_TO_VEHICLES_FOLDER[0]

    onlyfiles = [f for f in listdir(PATH_TO_VEHICLES_FOLDER) if isfile(
        join(PATH_TO_VEHICLES_FOLDER, f)) and "png" in f]
    # assert len(onlyfiles) == NUM_FILES, "Not the full VEDAI 1024 Dataset"
    rgb = np.zeros((len(subset), X_PIXELS, Y_PIXELS, 3))
    infra = np.zeros((len(subset), X_PIXELS, Y_PIXELS, 1))
    indices = subset

    print(indices)
    print(rgb.shape)
    index_filename_map = {}
    im_cnt = 0

    for index in indices:
        pair = [file for file in onlyfiles if str(index) in file]
        if pair:
            for file in pair:
                index_filename_map[im_cnt] = file.split('_')[0]
                im = imageio.imread(PATH_TO_VEHICLES_FOLDER + '/' + file)
                if "co" in file:
                    # print('Inserting RGB @ '+ str(im_cnt))
                    rgb[im_cnt, :, :, :] = np.reshape(
                        im, (tuple([1]) + im.shape))
                elif "ir" in file:
                    # print('Inserting Infra @ '+ str(im_cnt))
                    infra[im_cnt, :, :, :] = np.reshape(
                        im, (tuple([1]) + im.shape + tuple([1])))
            im_cnt = im_cnt + 1
        else:
            print("The following image is missing!: " + index)
    # print(index_filename_map)
    f = open(PATH_TO_VEHICLES_FOLDER + "_mapping.txt", "w")
    f.write(str(index_filename_map))
    f.close()
    return rgb, infra


def scan_dataset(PATH_TO_VEHICLES_FOLDER):
    # Takes in the full path to the unzipped "VEHICULES" folder
    # Returns a list of all the files
    # and saves a dataset summary text file with list of all file names
    MAX_INDEX = 10
    PATH_TO_VEHICLES_FOLDER = PATH_TO_VEHICLES_FOLDER[0]
    indices = [format(n, '08') for n in range(MAX_INDEX)]
    export_files = []

    onlyfiles = [f for f in listdir(PATH_TO_VEHICLES_FOLDER) if isfile(
        join(PATH_TO_VEHICLES_FOLDER, f)) and "png" in f]

    for index in indices:
        pair = [file for file in onlyfiles if str(index) in file]
        if pair:
            for file in pair:
                if "co" in file:
                    export_files.append(file.split('_')[0])
        else:
            print("The following image is missing!: " + index)
    np.savetxt(PATH_TO_VEHICLES_FOLDER + '_summary.txt',
               export_files, delimiter=" ", fmt="%s")
    return export_files


def create_subsets(imgs, output_path, use_validation=True,
                   training_percent=0.7, testing_percent=0.3, SEED=1):
    # Takes a list of image file names and shuffles
    # them before splitting them into required subsets
    # Saves txt files containing the names of the files
    # used in each subset, no return value
    assert training_percent + \
        testing_percent == 1, "Training + testing percents must equal 1."
    random.seed(SEED)
    random.shuffle(imgs)
    print('Using ' + str(len(imgs)) + ' images.')
    print('Saving files to ' + output_path)
    if not use_validation:
        training_imgs = imgs[:int(len(imgs) * training_percent)]
        testing_imgs = imgs[int(len(imgs) * training_percent):]
        np.savetxt(output_path + 'training.txt',
                   training_imgs, delimiter=" ", fmt="%s")
        np.savetxt(output_path + 'testing.txt',
                   testing_imgs, delimiter=" ", fmt="%s")
        return training_imgs, testing_imgs
    else:
        validation_split = 0.3  # use 30% of training dataset for validation
        training_imgs = imgs[int(len(imgs) * validation_split *
                                 training_percent):int(len(imgs) *
                                                       training_percent)]
        validation_imgs = imgs[:int(
            len(imgs) * validation_split * training_percent)]
        testing_imgs = imgs[int(len(imgs) * training_percent):]
        np.savetxt(output_path + 'validation.txt',
                   validation_imgs, delimiter=" ", fmt="%s")
        np.savetxt(output_path + 'training.txt',
                   testing_imgs, delimiter=" ", fmt="%s")
        np.savetxt(output_path + 'testing.txt',
                   testing_imgs, delimiter=" ", fmt="%s")
        return training_imgs, validation_imgs, testing_imgs


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


def downsample_image(image, block=(4, 4, 1)):
    # Downsamples numpy array image by factor
    # Returns  the downsampled copy
    return downscale_local_mean(image, block)


def reconstruct_patches(patches, image_size):
    # TODO: Create a function which reconstructs an image
    # when given patches created by non_overlapping_patches
    # Discards predictions for zero border
    pass


class VEDAISequence(Sequence):

    def __init__(self, rgb, infra, ims_per_batch):
        self.r, self.i = rgb, infra
        self.ims_per_batch = ims_per_batch

    def __len__(self):
        # Returns number of batches given training set and ims_per_batch
        return int(np.ceil(len(self.r) / float(self.ims_per_batch)))

    def __getitem__(self, idx):
        # Number of patches * ims_per_batch
        batchsz = 256 * self.ims_per_batch
        # RGB and infra
        channels = 4
        # Default patch size
        patch_x, patch_y = 64, 64

        # Batch number * ims_per_batch
        start = idx * self.ims_per_batch
        # Batch number * ims_per_batch  + 1
        end = (idx + 1) * self.ims_per_batch

        # Preallocate arrays of the correct size
        high_res = np.zeros((batchsz, patch_x, patch_y, channels))
        im_num = 0
        for ind in range(start, end):
            st, stp = im_num * 256, (im_num + 1) * 256
            im_num += 1
            rgb = imageio.imread(self.r[ind])
            infra = imageio.imread(self.i[ind])
            high_res[st:stp, :, :, :] = non_overlapping_patches(
                combine_rgb_infra(rgb, infra))
        low_res = np.asarray([downsample_image(patch) for patch in high_res])
        return low_res, high_res

import numpy as np
import tensorflow as tf

from os import listdir
from os.path import isfile, join
from skimage.transform import downscale_local_mean
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import random
import imageio


def normalize(image):
    # Normalizes images between -1 and 1
    return (image - 127.5) / 127.5


def normalize01(image):
    # Normalizes images between 0 and 1
    return image / 255


def un_normalize(image):
    # Takes an image on [-1,1] and scales it to [0,1]
    return (image + 1) / 2


def read_VEDAI(subset, PATH_TO_VEHICLES_FOLDER, filename):
    # Takes in the full path to the unzipped "VEHICULES" folder
    # Returns mapping dict, RGB and Infrared images in
    # (images, x, y, channels) format,
    # saves a txt file with mapping of rgb/infra idx to filename
    # NUM_FILES = 2536
    # MAX_INDEX = 1272
    X_PIXELS = 1024
    Y_PIXELS = 1024
    PATH_TO_VEHICLES_FOLDER = PATH_TO_VEHICLES_FOLDER  # [0]

    onlyfiles = [f for f in listdir(PATH_TO_VEHICLES_FOLDER) if isfile(
        join(PATH_TO_VEHICLES_FOLDER, f)) and "png" in f]
    rgb = np.zeros((len(subset), X_PIXELS, Y_PIXELS, 3))
    infra = np.zeros((len(subset), X_PIXELS, Y_PIXELS, 1))
    indices = subset

    index_filename_map = {}
    im_cnt = 0

    for index in indices:
        pair = [file for file in onlyfiles if str(index) in file]
        if pair:
            for file in pair:
                index_filename_map[im_cnt] = file.split('_')[0]
                im = imageio.imread(PATH_TO_VEHICLES_FOLDER + '/' + file)
                if "co" in file:
                    rgb[im_cnt, :, :, :] = np.reshape(
                        im, (tuple([1]) + im.shape))
                elif "ir" in file:
                    infra[im_cnt, :, :, :] = np.reshape(
                        im, (tuple([1]) + im.shape + tuple([1])))
            im_cnt = im_cnt + 1
        else:
            print("The following image is missing!: " + index)
    f = open(local_dir + "_mapping.txt", "w")
    f.write(str(index_filename_map))
    f.close()
    return rgb, infra


def scan_dataset(data_path, output_path, number_of_imgs):
    # Takes in the full path to the unzipped "VEHICULES" folder
    # Returns a list of all the files
    # and saves a dataset summary text file with list of all file names
    MAX_INDEX = number_of_imgs
    indices = [format(n, '08') for n in range(MAX_INDEX)]
    export_files = []

    onlyfiles = [f for f in listdir(data_path) if isfile(
        join(data_path, f)) and "png" in f]

    for index in indices:
        pair = [file for file in onlyfiles if str(index) in file]
        if pair:
            for file in pair:
                if "co" in file:
                    export_files.append(file.split('_')[0])
        else:
            print("The following image is missing!: " + index)
    np.savetxt(output_path + '_summary.txt',
               export_files, delimiter=" ", fmt="%s")

    return export_files


def create_subsets(img_list, output_path, use_validation=True,
                   training_percent=0.7, testing_percent=0.3, SEED=1):
    # Takes a list of image file names and shuffles
    # them before splitting them into required subsets
    # Saves txt files containing the names of the files
    # used in each subset, no return value

    assert training_percent + \
        testing_percent == 1, "Training + testing percents must equal 1."
    random.seed(SEED)
    random.shuffle(img_list)
    print('Using ' + str(len(img_list)) + ' images.')
    print('Saving files to ' + output_path)
    if not use_validation:
        training_imgs = img_list[:int(len(img_list) * training_percent)]
        testing_imgs = img_list[int(len(img_list) * training_percent):]
        np.savetxt(output_path + 'training.txt',
                   training_imgs, delimiter=" ", fmt="%s")
        np.savetxt(output_path + 'testing.txt',
                   testing_imgs, delimiter=" ", fmt="%s")
        return training_imgs, testing_imgs
    else:
        validation_split = 0.3  # use 30% of training dataset for validation
        training_imgs = img_list[int(len(img_list) * validation_split *
                                 training_percent):int(len(img_list) *
                                                       training_percent)]
        validation_imgs = img_list[:int(
            len(imgs) * validation_split * training_percent)]
        testing_imgs = img_list[int(len(img_list) * training_percent):]
        np.savetxt(output_path + 'validation.txt',
                   validation_imgs, delimiter=" ", fmt="%s")
        np.savetxt(output_path + 'training.txt',
                   testing_imgs, delimiter=" ", fmt="%s")
        np.savetxt(output_path + 'testing.txt',
                   testing_imgs, delimiter=" ", fmt="%s")
        return training_imgs, validation_imgs, testing_imgs


def data_explore(data):
    print("Shape of the data is" + str(data.shape))
    print("Dtype of the data is" + str(data.dtype))


def combine_rgb_infra(rgb, infra):
    # Concatenates the two modalities along the channels axis
    four_channel = np.concatenate((rgb, infra), axis=-1)
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
    if image.ndim == 4:
        block = (1, 4, 4, 1)
    return downscale_local_mean(image, block)


def restitch_image_patches(patches, img_dim=(1024, 1024, 4)):
    patch_shape = patches.shape
    w_patches = patch_shape[1]
    assert patch_shape[1] == patch_shape[2], 'Expecting square patches, w = h'
    h_patches = w_patches
    c_patches = patch_shape[3]

    patch_per_w = int(img_dim[0] / w_patches)

    img_stitched = np.zeros(shape=img_dim)

    for c in range(c_patches):
        i = 0  # Patch indexer
        for ih in range(patch_per_w):  # Height
            y_coord_1 = ih * h_patches
            y_coord_2 = y_coord_1 + h_patches
            for iw in range(patch_per_w):  # Width
                x_coord_1 = iw * w_patches
                x_coord_2 = x_coord_1 + w_patches

                img_stitched[y_coord_1:y_coord_2,
                             x_coord_1:x_coord_2, c] = patches[i, :, :, c]
                i = i + 1

    return img_stitched


def reconstruct_patches(patches, image_size=(1024, 1024)):
        # Reconstructs an image
        # when given patches created by overlapping_patches
    return reconstruct_from_patches_2d(patches, image_size)


def get_images_to_four_chan(img_name, DATASET_PATH, ch_num=4):
    co = imageio.imread(DATASET_PATH + img_name + '_co.png')
    if ch_num == 4:
        ir = imageio.imread(DATASET_PATH + img_name + '_ir.png')
        rgb = np.reshape(co, (tuple([1]) + co.shape))
        infra = np.reshape(ir, (tuple([1]) + ir.shape + tuple([1])))
        return combine_rgb_infra(rgb, infra)
    elif ch_num == 3:
        return np.reshape(co, (tuple([1]) + co.shape))


def load_data_vehicles(DATASET_PATH, num_images, rgb=False,
                       scale01=False, img_spec=None):
        # get all files from the directory
    onlyfiles = [f for f in listdir(DATASET_PATH) if isfile(
        join(DATASET_PATH, f)) and "png" in f]

    if img_spec is not None:
        assert img_spec < len(
            onlyfiles), "Image not found. Pick a smaller number."
        tmp_path = onlyfiles[img_spec]
        onlyfiles = list([tmp_path])
        # onlyfiles = []
        # onlyfiles.append(tmp_path)
        print("using single image " + str(onlyfiles))
    else:
        assert num_images < len(
            onlyfiles), "Too many images. Pick a smaller number."
        onlyfiles = onlyfiles[0:num_images]
        print("using {0} images".format(num_images))

    channels = 3     # RGB
    patch_x, patch_y = 64, 64  # patch size

    # Preallocate array of the correct size
    imgs_hr = np.zeros((len(onlyfiles), patch_x, patch_y, channels))

    print(len(onlyfiles))

    img_idx = 0
    for i in range(len(onlyfiles)):
        co = imageio.imread(DATASET_PATH + onlyfiles[i])
        rgb = np.reshape(co, (tuple([1]) + co.shape))
        if rgb.shape == (1, 64, 64, 3):
            if scale01:
                imgs_hr[img_idx, :, :, :] = normalize01(rgb)
            else:
                imgs_hr[img_idx, :, :, :] = normalize(rgb)
            img_idx += 1
    imgs_lr = np.asarray([downsample_image(patch) for patch in imgs_hr])
    return imgs_hr, imgs_lr

'''
Load patched data into memory (highres/lowres) patches
args:
----
    data_idx:     The index of current batch in overall data
    idx_file:     list containing files to patch
    data_path:    Path to actual 1024x1024 images 
    scale01:      Scale images to [0,1] otherwise scales to [-1,1]
    batch_size:   Number of 1024x1024 images to patch per batch
'''
def load_data(data_idx, img_list, data_path, scale01=False, batch_size=1):
    # read in batch of file names from txt file with randomized filenames
    # return the lr and hr patches

    # read x lines from txt file
    #text_file = open(idx_file, "r")
    #img_files = text_file.read().strip().split('\n')
    #text_file.close()
    
    # Number of patches * ims_per_batch
    batchsz = 256 * batch_size
    # RGB
    channels = 4
    # Default patch size
    patch_x, patch_y = 64, 64

    # Preallocate arrays of the correct size
    imgs_hr = np.zeros((batchsz, patch_x, patch_y, channels))

    # Batch number * ims_per_batch
    start = data_idx
    end = data_idx + batch_size
    
    im_num = 0
    for i in range(start, end):
        st, stp = im_num * 256, (im_num + 1) * 256
        im_num += 1
        img = get_images_to_four_chan(img_list[i], data_path, channels)
        if scale01:
            img = normalize01(img)
        else:
            img = normalize(img)
        patch = overlapping_patches(img)
        imgs_hr[st:stp, :, :, :] = patch

    imgs_lr = np.asarray([downsample_image(patch) for patch in imgs_hr])

    data_idx = data_idx + batch_size  # update current file_idx
    return imgs_hr, imgs_lr, data_idx


def find_vehicles(channels=3, patch_size=64):
    # read x lines from txt file
    dir_path = '../data/'
    IM_DIM = 1023
    text_file = open(dir_path + "annotation1024.txt", "r")
    vehicles = text_file.read().split('\n')
    text_file.close()

    vehicles_dict = {}
    dict_idx = 0
    for v in vehicles:
        line_items = v.split()
        if (len(line_items) > 3):
            vehicles_dict[dict_idx] = (line_items[0], float(
                line_items[1]), float(line_items[2]))
            dict_idx = dict_idx + 1

    print(len(vehicles_dict))
    items = list(vehicles_dict.items())
    random.shuffle(items)
    for key, val in items:
        # load image from dirimport os
        exists = isfile(
            dir_path + 'Vehicules1024/' + val[0] + '_co.png')
        if exists and len(val[0]) > 1:
            img = get_images_to_four_chan(val[0], dir_path, channels)
            # get vehicle patch with patch_size
            xa = int(val[1] - patch_size // 2)
            xb = int(val[1] + patch_size // 2)
            ya = int(val[2] - patch_size // 2)
            yb = int(val[2] + patch_size // 2)
            if xa < 0:  # patch is out of bounds
                # subtract the negative (add) to shift back into image
                xa = xa - xa
            if xb > IM_DIM:
                xb = xb - (xb - IM_DIM)
            if ya <= 0:  # patch is out of bounds
                # subtract the negative (add) to shift back into image
                ya = ya - ya
            if xb > IM_DIM:
                yb = yb - (yb - IM_DIM)
            patch = img[:, xa:xb, ya:yb, :].squeeze()
            imageio.imwrite(dir_path + 'vehicle_patches_64/' +
                            val[0] + "_" + str(key) + '.png', patch)


def get_img_patches(im_hr, im_lr, img_idx=0, patch_size=64, img_size=1024):

    PATCH_PER_IMG = (img_size / patch_size)**2

    begin_idx = int(img_idx * PATCH_PER_IMG)
    end_idx = int(begin_idx + PATCH_PER_IMG)

    im_hr_patched = im_hr[begin_idx:end_idx, :, :, :]
    im_lr_patched = im_lr[begin_idx:end_idx, :, :, :]

    return im_hr_patched, im_lr_patched

def PSNR(im1, im2): 
    assert im1.shape == im2.shape, 'Images must be the same dimension'
    im_h = im1.shape[0]
    im_w = im1.shape[1]
    im_c = im1.shape[2]
    mse = np.sum((im1 - im2)**2)/(im_w*im_h*im_c)
    return 10*np.log10(1**2/mse)
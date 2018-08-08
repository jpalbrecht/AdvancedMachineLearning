# final project

# skimage
from skimage import io
from skimage import filters
from skimage import color
from skimage import morphology

# sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# numpy
import numpy as np

# plotting
from matplotlib.pyplot import imshow, subplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import save_model
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Convolution2D
from keras.layers import Flatten, MaxPooling2D
from keras.utils import to_categorical

# general
import os
from sys import stdout
import pickle
import random

# file processing
from shutil import copyfile
from shutil import rmtree


def split_img_set(load_path_l, test_path_l, train_path_l, test_percentage=0.1):
    """
    Function to copy all files in load_path_l in either test_path_l folder or train_path_l. test_percentage decides how
    many random files will be copied to test_path_l.

    Inputs:
        load_path_l         - path to folder which files should be split in two folders
        test_path_l         - path to folder where to copy test files to
        train_path_l        - path to folder where to train test files to
        test_percentage     - percentage of files which should be randomly chosen to be copied to test_path_l

    Output:
        None
    """
    # list all files in path and take random sample as test dataset
    img_list = os.listdir(load_path_l)
    rand_smpl = random.sample(range(len(img_list)), int(np.ceil(float(len(img_list)) * test_percentage)))
    # delete test_path and train_path and create them again
    if os.path.exists(test_path_l):
        rmtree(test_path_l, ignore_errors=True)
        os.mkdir(test_path_l)
    else:
        os.mkdir(test_path_l)
    if os.path.exists(train_path_l):
        rmtree(train_path_l, ignore_errors=True)
        os.mkdir(train_path_l)
    else:
        os.mkdir(train_path_l)
    # copy picture to its corresponding path, either testing or training
    for idx, img_name in enumerate(img_list):
        # if idx occurs in rand_smpl copy picture to testing path else to testing path
        if idx in rand_smpl:
            copyfile(load_path_l + img_name, test_path_l + img_name)
        else:
            copyfile(load_path_l + img_name, train_path_l + img_name)


def preprocess_pictures(img_rgba_l, debug=False, pdf_page=None):
    """
    Function to preprocess 4 channel png to a binary image

    Inputs:
        img_rgba_l         - image with four channels
        debug              - flag indicating to print debug plots of procedure
        pdf_page           - PdfPages object, all plots will be written to pdf

    Output:
        img_bin_morph_l       - binary image
    """
    # ## convert ###
    # convert to rgb
    img_rgb = color.rgba2rgb(img_rgba_l)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_rgb)
        if isinstance(pdf_page, PdfPages):
            pdf_page.savefig(fig)

    # convert to one channel
    img_grey = color.rgb2grey(img_rgb)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_grey, cmap='gray')
        if isinstance(pdf_page, PdfPages):
            pdf_page.savefig(fig)

    # ## filter ###
    # apply otsu filtering
    otsu_val = filters.threshold_otsu(img_grey)
    img_bin = img_grey < otsu_val
    # debug mode
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_bin, cmap='gray')
        if isinstance(pdf_page, PdfPages):
            pdf_page.savefig(fig)

    # apply a morphological opening and erosion
    img_bin_morph_l = morphology.binary_opening(img_bin)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_bin_morph_l, cmap='gray')
        if isinstance(pdf_page, PdfPages):
            pdf_page.savefig(fig)

    img_bin_morph_l = morphology.binary_erosion(img_bin_morph_l)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_bin_morph_l, cmap='gray')
        if isinstance(pdf_page, PdfPages):
            pdf_page.savefig(fig)

    return img_bin_morph_l


def crop_pictures(img_bin_morph_l, thresh_pixel, debug=False, pdf_page=None):
    """
    Function to crop image according to its column sum. Crops image according to a given threshold.

    Inputs:
        img_bin_morph_l         - binary image
        thresh_pixel            - threshold where to start cropping (amount of pixels needed in a column)
        debug                   - flag indicating to print debug plots of procedure
        pdf_page                - PdfPages object, all plots will be written to pdf


    Output:
        img_bin_morph_crop_l       - cropped img_bin_morph_l
    """
    # ## cropping ###
    # the idea is to get the column sums and decide whether there are enough pixels belonging to a letter in each column

    # get shape of picture,  column-sum over picture matrix
    img_dim = img_bin_morph_l.shape
    colsum = np.zeros(img_dim[1])
    for i in range(0, img_dim[1]):
        for j in range(0, img_dim[0]):
            colsum[i] = colsum[i] + img_bin_morph_l[j][i]

    # crop image
    # where do we need to crop? where there are less than threshold pixel in a column
    start = np.argmax(colsum > thresh_pixel)
    rowsum_rev = colsum[::-1]
    stop = img_dim[1] - np.argmax(rowsum_rev > thresh_pixel)
    img_bin_morph_crop_l = img_bin_morph_l[:, start:stop]

    # debug mode:
    if debug:
        fig = plt.figure()
        fig2 = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = fig2.add_subplot(111)
        ax.plot(colsum)
        ax2.imshow(img_bin_morph_crop_l, cmap='gray')
        if isinstance(pdf_page, PdfPages):
            pdf_page.savefig(fig)
            pdf_page.savefig(fig2)

    # return cropped img
    return img_bin_morph_crop_l


def extract_pixel(img_bin_morph_crop_l):
    """
    Function to extract all pixels set to "True" in binary image

    Inputs:
        img_bin_morph_crop_l    - binary image

    Output:
        indices_l       - all images defining the picture (e.g. set to true)
    """
    # ## extracting ###
    # get all indices of TRUE values. These are the points that belong to letters
    indices_l = []
    img_dim_crop = img_bin_morph_crop_l.shape
    for i in range(img_dim_crop[0]):
        for j in range(img_dim_crop[1]):
            if img_bin_morph_crop_l[i][j]:
                indices_l.append([i, j])

    return np.matrix(indices_l)


def get_center_pos(indices_l, n_cluster=5):
    """
    Function to get the initial center positions for clustering from any picture. Binary image given as indices
    which are set to "True"

    Inputs:
        indices_l    - indices of y and x-axis which are set to True in any binary image
        n_cluster    - number of clusters (e.g. letters) to build

    Output:
        kmeans       - keras kmeans result
    """
    cluster_pos = np.empty((n_cluster, 2), dtype=int)
    # get indices of x-axes
    ind_x = np.array(indices_l[:, 1])
    ind_x_min = np.min(ind_x)
    ind_x_max = np.max(ind_x)
    # get indices of y-axes
    ind_y = np.array(indices_l[:, 0])
    ind_y_min = np.min(ind_y)
    ind_y_max = np.max(ind_y)
    # interval size x
    interval_size_x = ind_x_max - ind_x_min
    # interval size y
    interval_size_y = ind_y_max - ind_y_min
    cluster_interval = np.floor(float(interval_size_x)/(2*n_cluster))
    for idx, clust in enumerate(range(1, 2*n_cluster, 2)):
        cluster_pos[idx, 1] = (clust*cluster_interval + ind_x_min)
        cluster_pos[idx, 0] = (interval_size_y/2)

    return cluster_pos


def cluster_image(indices_l, debug=False, pdf_page=None):
    """
    Function to get the dictionary linking from cluster-position to label and backwards.

    Inputs:
        indices_l    - indices of y and x-axis of picture which are set to True in any binary image
        debug        - flag indicating to print debug plots of procedure
        pdf_page     - PdfPages object, all plots will be written to pdf


    Output:
        kmeans         - keras kmeans result
    """
    # ## clustering ###
    cluster_start_pos = get_center_pos(indices_l, n_cluster=5)
    kmeans = KMeans(n_clusters=5, n_init=1, max_iter=300, init=cluster_start_pos).fit(indices_l)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter_x = np.array(indices_l[:, 0])
        scatter_y = np.array(indices_l[:, 1])[::-1]
        cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'black', 4: 'yellow'}
        for g in np.unique(kmeans.labels_):
            idx = np.where(kmeans.labels_[::-1] == g)
            ax.scatter(scatter_y[idx], scatter_x[idx], c=cdict[g], label=g)
            ax.legend()
        if isinstance(pdf_page, PdfPages):
            pdf_page.savefig(fig)

    return kmeans


def get_label_ind(img_kmeans_l):
    """
    Function to get the dictionary linking from cluster-position to label and backwards.

    Inputs:
        img_kmeans_l    - keras kmeans result

    Output:
        label_ind_l         - list linking from position of label to cluster-position
        cluster_ind_l       - dictionary linking from cluster-position to label
    """
    # ## get labeling ###
    # extract label from picture name
    # sort cluster centers acc. to x-dimension of img
    centers = img_kmeans_l.cluster_centers_

    x_centers = []
    for i in range(0, len(centers)):
        x_centers.append(centers[i][1])

    # example:
    # name = 2gyb6
    # label_ind[3] = 0
    # sort example: the 3 entry of the name (b) has label 0
    label_ind_l = sorted(range(len(x_centers)), key=lambda k: x_centers[k])

    # create backwards dict for later reverse lookup
    cluster_ind_l = dict()
    for idx in range(len(x_centers)):
        cluster_ind_l[label_ind_l[idx]] = idx

    return label_ind_l, cluster_ind_l


def shift_x_axis(label_ind_x_l, new_mean):
    """
    Function to extract all letters in given image and save them separately.

    Inputs:
        label_ind_x_l    - all indices from x-axis of any picture
        new_mean         - integer, specifying were to shift all points in label_ind_x_l to

    Output:
        label_ind_x_shift   - shifted x-axis picture
    """
    # we will shift all points to the center of a mean.
    label_ind_x_max = float(np.max(label_ind_x_l))
    label_ind_x_min = float(np.min(label_ind_x_l))
    label_ind_x_mean = np.ceil(((label_ind_x_max - label_ind_x_min) / 2) + label_ind_x_min)
    # filter for error:
    # if the shifted mean is bigger than the mean we want to scale our picture to we have a problem
    # since some values will end with negative indices!
    if (label_ind_x_mean - label_ind_x_min) >= new_mean:
        # sub-picture can not be processed
        return 0

    # empty new x-coord vector
    label_ind_x_shift = []
    for pnt in label_ind_x_l:
        # calculate difference to tmp_ind_x_mean
        shift = pnt - label_ind_x_mean
        # create a new point with same shift from new mean
        label_ind_x_shift.append(new_mean + shift)

    return label_ind_x_shift


def subpicturing(img_bin_morph_crop_l, img_name_l, indices_l, kmeans_l, debug=False, pdf_page=None, train=True):
    """
    Function to extract all letters in given image and save them separately.

    Inputs:
        img_bin_morph_crop_l    - binary image
        img_name_l              - name of the image, must contain all labels of each letter in
                                        img_bin_morph_crop_l in correct order
        indices_l               - all pixels in binary image that are "True"
        kmeans_l                - results of keras kmeans algorithm
        debug                   - flag indicating to print debug plots of procedure
        pdf_page                - PdfPages object, all plots will be written to pdf
        train                   - flag, indicating whether the label of  image should be derived out of its name or not
                                        i.e. for training procedure label-processing is necessary!

    Output:
        sub_img_list_l      - list, containing all isolated letters in img_bin_morph_crop_l as images themself
        label_names         - label to all isolated letter images in sub_img_list_l
        cluster_ind         - dictionary linking from cluster-position in img_bin_morph_crop_l to letter in label_names,
                                    since kmeans algorithm does not "read" letters in
                                    img_bin_morph_crop_l from left to right !
    """
    # ## sub-picturing ###
    dim_img_bmc = img_bin_morph_crop_l.shape
    # get the label_indices for later lookup of the subpictures-label
    label_ind, cluster_ind = get_label_ind(kmeans_l)

    # for each label produce its own picture since it is hopefully an own letter
    kmean_label = np.unique(kmeans_l.labels_)
    # list of pictures
    sub_img_list_l = []
    label_names = []
    for label in kmean_label:
        # get all indices belonging to that label
        label_filter_bool = kmeans_l.labels_ == label
        label_indices = indices_l[label_filter_bool]
        label_ind_y = label_indices[:, 0]
        label_ind_x = label_indices[:, 1]

        # use the y axis of our pictures as orientation - thus our sub-picture will have the mean of y-dimension/2
        label_ind_x_shift = shift_x_axis(label_ind_x, np.floor(dim_img_bmc[0] / 2))
        if label_ind_x_shift == 0:
            print("picture " + img_name_l + " can not be processed!")
            break

        # extract label from image name and append to list if we know the image-name got the labels
        if train:
            label_names.append(img_name_l[label_ind.index(label)])

        # create empty pic of y-dimension (of our preprocessed picture) width and length and fill with pixels
        letter_pic = np.zeros((dim_img_bmc[0], dim_img_bmc[0]))
        # fill empty pic with pixels
        for i in range(len(label_ind_y)):
            letter_pic[int(label_ind_y[i])][int(label_ind_x_shift[i])] = 1

        # debug mode
        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(letter_pic, cmap='gray')
            if isinstance(pdf_page, PdfPages):
                pdf_page.savefig(fig)

        # return sub-pictures with corresponding labeling
        sub_img_list_l.append(letter_pic)

    return sub_img_list_l, label_names, cluster_ind


def preprocess_data(load_path, save_path="",
                    extension=".png", load_data=False, verbose=True, debug=False, pdf_page=None, train=True):
    """
    Function to execute the preprocessing procedure for pictures in given path.

    Inputs:
        load_path        - path to the folder containing the images which need to be preprocessed
        save_path        - path where to store preprocessed pictures
        extension        - string indicating which pictures to load, currently allowed: .png
        load_data        - flag indicating whether to load the data from load_path instead
                                of doing preprocessing procedure. If set to true folder specified in load_path must
                                contain: processed_img_list label_img_list label_dict_list files!
        verbose         - flag, indicating wheather to print additionallyinformations
        debug           - flag indicating to print debug plots of procedure
        pdf_page        - PdfPages object, all plots will be written to pdf
        train           - flag, indicating whether the label of the image should be derived out of its name or not
                                i.e. for training procedure label-processing is necessary!

    Output:
        processed_img_list_l        - list with preprocessed img(s)
        label_img_list_l            - label of each image in processed_img_list_l
        cluster_dict_list_l       - list with number_transform copies of each image
    """
    # either we load the preprocessed data or we do the preprocessing
    if load_data:
        with open(load_path + "label_img_list", 'rb') as pickle_dumb2:
            label_img_list_l = pickle.load(pickle_dumb2)
        with open(load_path + "processed_img_list", "rb") as pickle_dumb:
            processed_img_list_l = pickle.load(pickle_dumb)
        return processed_img_list_l, label_img_list_l
    else:
        if os.path.isfile(load_path):
            load_path, name = os.path.split(load_path)
            img_list = [name]
        else:
            # list of all pictures in folder under "path"
            img_list = os.listdir(load_path)

        # define a huge datasets
        processed_img_list_l = []
        label_img_list_l = []
        cluster_dict_list_l = []

        # list over all elements
        for img_idx, img_name in enumerate(img_list):
            # get filename and extension
            filename, file_extension = os.path.splitext(img_name)

            # load file
            if file_extension == extension:
                img_rgba = io.imread(os.path.join(load_path, filename + file_extension))
            else:
                print("File " + filename + " can not be processed. Processing only "
                      + extension + " files! Got " + file_extension)
                continue

            # check if correctly loaded and process
            if type(img_rgba) is np.ndarray:
                # show info
                if verbose:
                    print(str(img_idx) + ": " + filename)
                # processing of image
                img_bin_morph = preprocess_pictures(img_rgba, pdf_page=pdf_page, debug=debug)
                img_bin_morph_crop = crop_pictures(img_bin_morph, 4, pdf_page=pdf_page, debug=debug)
                indices = extract_pixel(img_bin_morph_crop)
                img_kmeans = cluster_image(indices, pdf_page=pdf_page, debug=debug)
                sub_img_list, label_name, cluster_dict = subpicturing(img_bin_morph_crop,
                                                                      filename, indices,
                                                                      img_kmeans, debug, pdf_page, train=train)
                # add to one BIG set
                for i in range(len(sub_img_list)):
                    processed_img_list_l.append(sub_img_list[i])
                    if train:
                        label_img_list_l.append(label_name[i])
                cluster_dict_list_l.append(cluster_dict)

        # save img_list
        if not save_path == "":
            with open(save_path + "processed_img_list", "wb+") as fp:
                pickle.dump(processed_img_list_l, fp)

            # save label_list
            with open(save_path + "label_img_list", "wb+") as fp:
                pickle.dump(label_img_list_l, fp)

            # save label dictionary
            with open(save_path + "label_dict_list", "wb+") as fp:
                pickle.dump(cluster_dict_list_l, fp)

        return processed_img_list_l, label_img_list_l, cluster_dict_list_l


def augment_data(processed_img_list_l,
                 label_img_list_l, image_data_generator, number_transform, debug=False, pdf_page=None):
    """
    Function to copy images in processed_img_list_l with slightly derivations.

    Inputs:
        processed_img_list_l        - img(s) which need to be duplicated
        label_img_list_l            - label of the img(s) to copy, same length as processed_img_list_l
        image_data_generator        - ImageDataGenerator object, specifying how to alter pictures
        number_transform            - number of times image is copied
        debug                       - flag indicating to print debug plots of procedure
        pdf_page                    - PdfPages object, all plots will be written to pdf

    Output:
        pro_aug_img_list_l     - list with number_transform copies of each image
        label_aug_img_list_l   - label for each image in pro_aug_img_list_l

    """

    # empty lists
    pro_aug_img_list_l = []
    label_aug_img_list_l = []
    # transform each image "number_transform" times and save label and transformed image
    for idx, img_l in enumerate(processed_img_list_l):
        for iter_transform in range(number_transform):
            img_dim = img_l.shape
            img_aug = image_data_generator.random_transform(img_l.reshape((img_dim[0], img_dim[1], 1)))

            # append to new dataframe
            pro_aug_img_list_l.append(img_aug)
            label_aug_img_list_l.append(label_img_list_l[idx])

            # show the first 16 pictures
            if debug and iter_transform < 16:
                fig = subplot(4, 4, iter_transform + 1)
                imshow(img_aug.reshape((img_dim[0], img_dim[1])), cmap='gray')
                if isinstance(pdf_page, PdfPages):
                    pdf_page.savefig(fig)

    return pro_aug_img_list_l, label_aug_img_list_l


def label_to_categorical(label_aug_img_list_l, info=False):
    """
    Function to convert the label given as a list onto categorical numpy array.

    Inputs:
        label_aug_img_list_l        - list containing all labels as real letters
        info                        - flag, indicating to print info about the labels in the list

    Output:
        label_aug_img_arr_cat_l     - nparray, containing all labels as in input specified in the same order converted
                                        to categorical
        label_aug_img_dict_l        - dictionary linking from categorical class to letter

    """
    # create empty list and dictionary
    label_aug_img_arr_cat_l = []
    label_aug_img_dict_l = dict()
    info_l = dict()

    # categorical number
    idx_dict = 0

    # fill translation-dictionary and build categorical list
    for symbol in label_aug_img_list_l:
        if symbol not in label_aug_img_dict_l:
            label_aug_img_dict_l[symbol] = idx_dict
            info_l[symbol] = 1
            idx_dict = idx_dict + 1
        else:
            info_l[symbol] = info_l[symbol] + 1
        label_aug_img_arr_cat_l.append(label_aug_img_dict_l[symbol])

    # pint occurence of each letter in dataset
    if info:
        print("Letter: occurence")
        for key_l in info_l.keys():
            print(key_l + ": " + str(info_l[key_l]))

    # convert to numpy array
    label_aug_img_arr_cat_l = to_categorical(np.asarray(label_aug_img_arr_cat_l, dtype=np.int), num_classes=idx_dict)

    return label_aug_img_arr_cat_l, label_aug_img_dict_l


def plot_hist(hist_l, pdf_page=None):
    """
    Function to plot training history.

    Inputs:
        hist_l          - training history of any model
        pdf_page        - PdfPages object, all plots will be written to pdf

    Output:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(hist_l['acc'], label='Training Accuracy')
    ax.plot(hist_l['val_acc'], label='Validation Accuracy')
    ax.legend(loc='lower right')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(hist_l['loss'], label='Training Loss')
    ax2.plot(hist_l['val_loss'], label='Validation Loss')
    ax2.legend()

    # plot to pdf?
    if isinstance(pdf_page, PdfPages):
        pdf_page.savefig(fig)
        pdf_page.savefig(fig2)


def train_model(paia_shape_l,
                x_train_l, x_val_l, y_train_l, y_val_l, num_classes_l, save_path="", save_model_flag=False):
    """
    Function to train a sequential model.

    Inputs:
        paia_shape_l        - shape of the input data for Convolution2D-layer
        x_train_l           - img(s) of the test dataset
        x_val_l             - img(s) of the validation dataset
        y_train_l           - label of the training dataset
        y_val_l             - label of the validation dataset
        num_classes_l       - number of classes (number of possible output classes)
        save_path           - path where to save the model to of save_model_flag true
        save_model_flag     - flag indicating to save the model

    Output:
        pred_list_char  - the prediction (not categorical, real letters) as list
    """
    # set up own model
    own_model_l = Sequential()
    # convolution layer & activation
    own_model_l.add(
        Convolution2D(16, (10, 10), activation='relu', input_shape=(paia_shape_l[1], paia_shape_l[2], paia_shape_l[3])))
    # pooling layer
    own_model_l.add(MaxPooling2D())
    # another conv
    # own_model.add(Convolution2D(16, (10, 10),
    #                             activation='relu',
    #                             input_shape=(paia_shape_l[1], paia_shape_l[2], paia_shape_l[3])))
    # pooling layer
    # own_model.add(MaxPooling2D())
    # flattering layer
    own_model_l.add(Flatten())
    # dense layer with 500 neuron
    own_model_l.add(Dense(500, activation='relu'))
    # softmax as the last activation layer to get a probability distribution for each class
    own_model_l.add(Dense(num_classes_l, activation='softmax'))
    # compile
    own_model_l.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # train on our data
    history_l = own_model_l.fit(x_train_l,
                                y_train_l,
                                batch_size=100,
                                epochs=6,
                                verbose=1,
                                validation_data=(x_val_l, y_val_l))

    # save model
    if save_model_flag:
        # save model
        save_model(own_model_l, save_path + "own_model.h5", True)
        # save training history
        with open(save_path + "own_model_history", "wb+") as fp:
            pickle.dump(history_l.history, fp)

    return own_model_l, history_l


def train_routine(data_path_l, res_path_l, test_save_path_l,
                  train_save_path_l, load_pictures=False, debug=False, save_model_flag=False, pdf_page=None):
    """
    Function to execute the training routine for given parameters.

    Inputs:
        data_path_l         - path to folder containing the images to process
        res_path_l          - path to folder to write results to
        test_save_path_l    - path to folder to copy test pictures to
        train_save_path_l   - path to folder to copy train pictures to
        load_pictures       - flag, indicating to not preprocess the pictures, instead load preprocessed pickle dump
                                if set, res_path_l must contain label_img_list and processed_img_list file
        debug               - flag indicating to print debug plots of processing procedure
        save_model_flag     - flag indicating to save the model
        pdf_page            - PdfPages object, all plots will be written to pdf

    Output:
        pred_list_char  - the prediction (not categorical, real letters) as list
    """

    # if we load the pictures we do not need to split them since they are already splitted
    if not load_pictures:
        print("split dataset ...")
        # split pictures randomly into two sets:
        split_img_set(data_path_l, test_save_path_l, train_save_path_l)

        print("preprocess pictures in: " + train_save_path_l)
        processed_img_list, label_img_list, label_dict_rew_list_l = preprocess_data(train_save_path_l,
                                                                                    res_path_l, ".png")
    else:
        print("load preprocess pictures from: " + res_path_l)
        processed_img_list, label_img_list, label_dict_rew_list_l = preprocess_data(res_path_l, load_data=True)

    # set up data generator
    img_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False)

    # show info for the labels:
    label_to_categorical(label_img_list, info=True)

    # augment data
    print("augment data ...")
    pro_aug_img_list, label_aug_img_list = augment_data(processed_img_list, label_img_list, img_gen, 16,
                                                        debug=debug, pdf_page=pdf_page)

    # convert data to np-array
    pro_aug_img_arr = np.asarray(pro_aug_img_list)
    # convert label to categorical
    label_aug_img_arr_cat, label_aug_img_dict = label_to_categorical(label_aug_img_list, info=False)

    # save dict for later back-translation
    if save_model_flag:
        with open(res_path_l + "own_model_class_dict", "wb+") as fp:
            pickle.dump(label_aug_img_dict, fp)

    # determine the number of classes we will do a training on
    num_classes = len(np.unique(label_aug_img_list))

    # determine shape of input data
    paia_shape = pro_aug_img_arr.shape

    # split data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(pro_aug_img_arr,
                                                        label_aug_img_arr_cat,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=label_aug_img_arr_cat)
    # split again, obtaining validation dataset
    val_set_size = 1000
    x_val = x_train[-val_set_size:]
    x_train = x_train[:-val_set_size]
    y_val = y_train[-val_set_size:]
    y_train = y_train[:-val_set_size]

    # call train routine
    print("train model ...")
    own_model_l, history_l = train_model(paia_shape, x_train, x_val, y_train,
                                         y_val, num_classes,  save_path=res_path, save_model_flag=save_model_flag)

    # check trainings hist
    plot_hist(history_l.history, None)
    # test model performance
    score = own_model_l.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return own_model_l, history_l, label_aug_img_dict


def get_pred_results(pred_list_l, label_dict, cluster_dict, verbose=True):
    """
    Function to predict translate between categorical prediction and real letter behind categorical.
    The prediction will be additionally sorted into the right order

    Inputs:
        pred_list_l         - the prediction as list as categorical
        label_dict          - dictionary linking from categorical class to letter
        cluster_dict        - dictionary giving the right order of the prediction in pred_list_l
        verbose             - flag indicating more output

    Output:
        pred_list_char  - the prediction (not categorical, real letters) as list
    """
    pred_list_char = np.empty(5, dtype=str)
    for idxs, pred in enumerate(pred_list_l):
        for key in label_dict.keys():
            if label_dict[key] == pred:
                pred_list_char[cluster_dict[idxs]] = key
    # print out results
    if verbose:
        for pred_char in pred_list_char:
            stdout.write(pred_char)
        stdout.write('\n')
        stdout.flush()

    return pred_list_char


def predict_single_img(model_l, img_path_l, label_dict_l, verbose=True, debug=False, pdf_page=None):
    """
    Function to predict a single image file given the model.

    Inputs:
        model_l         - path to folder containing: own_model.h5, own_model_history, own_model_class_dict
        img_path_l      - path to image file, must be png format, label for machine unknown
        label_dict_l    - dictionary linking from categorical class to letter
        verbose         - flag indicating more output
        debug           - flag indicating to print debug plots of processing procedure
        pdf_page        - PdfPages object, all plots will be written to pdf

    Output:
        pred_list_char  - the prediction (not categorical, real letters) as list
    """
    # preprocess single picture, not knowing the labels (train=False)
    proc_img_lst_l, label_img_lst_l, cluster_dict_lst_l = preprocess_data(img_path_l,
                                                                          load_data=False, verbose=verbose,
                                                                          debug=debug, pdf_page=pdf_page, train=False)
    # check if result not empty
    if not proc_img_lst_l:
        return []
    # reshape input for prediction
    proc_img_lst_reshape = []
    for sub_img in proc_img_lst_l:
        dim = sub_img.shape
        proc_img_lst_reshape.append(sub_img.reshape((dim[0], dim[1], 1)))
    # get prediction from model
    pred_list = model_l.predict_classes(np.asarray(proc_img_lst_reshape))
    # translate categoricals to characters and bring prediction in correct order
    if verbose:
        print("predicted for loaded image: ")
    pred_list_char = get_pred_results(pred_list, label_dict_l, cluster_dict_lst_l[0], verbose=verbose)

    return pred_list_char


def load_own_model(res_path_l):
    """
    Function to load an  previously trained model.

    Inputs:
        res_path_l          - path to folder containing: own_model.h5, own_model_history, own_model_class_dict

    Output:
        own_model_l         - keras model
        history_l           - training history
        class_dict_l        - dictionary, translating between categorical class and its corresponding letter
    """
    own_model_l = load_model(res_path_l + "own_model.h5")
    # load history
    with open(res_path_l + "own_model_history", 'rb') as pd:
        history_l = pickle.load(pd)
    # load class dict
    with open(res_path_l + "own_model_class_dict", 'rb') as pd2:
        class_dict_l = pickle.load(pd2)
    return own_model_l, history_l, class_dict_l


# ## MAIN ###
# Where are the captach pictures?
data_path = "captcha_dataset/samples/"
# Where should the results be written?
res_path = "results/"
# Where do you want to store the pictures used as test-set?
test_save_path = "captcha_dataset/test/"
# Where do you want to store the pictures used as train-set?
train_save_path = "captcha_dataset/train/"

# train model without loading preprocessed images
# own_model, history, own_model_class_dict = train_routine(data_path, res_path, test_save_path, train_save_path,
#                                                         load_pictures=False, save_model_flag=True)
# train model with loading preprocessed images
# train_routine(data_path, res_path, test_save_path, train_save_path, load_pictures=True,
# save_model_flag=True, debug=True)

# load already saved model
own_model, history, class_dict = load_own_model(res_path)

# create pdf output
pp = PdfPages(res_path + "plots.pdf")
plot_hist(history, pdf_page=pp)
res = predict_single_img(own_model, test_save_path + "6xpme.png", class_dict, verbose=True, debug=True, pdf_page=pp)
pp.close()

# predict all test img and compare prediction with name
img_name_list = os.listdir(test_save_path)
error = 0
correct = 0
for img in img_name_list:
    p = predict_single_img(own_model, test_save_path + img, class_dict, verbose=False, debug=False)
    # compare with name
    if ''.join(p) not in img:
        error = error + 1
    else:
        correct = correct + 1
print("Number of false prediction: " + str(error))
print("Number of correct prediction: " + str(correct))
print("Percentage of false prediction: " + str(round(float(error)/float(error + correct), 2)))

print("finished")

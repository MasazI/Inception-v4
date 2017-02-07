from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import h5py
import parmap
import argparse
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils
import cv2
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


def format_data(img_path, label, size):
    """
    Load img with opencv and reshape
    """
    img_color = cv2.imread(img_path)
    img_color = img_color[:, :, ::-1]
    img_color = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_AREA)
    img_color = img_color.reshape((1, size, size, 3))\
    #.transpose(0, 3, 1, 2)

    return img_color, label


def data_to_hdf5(data_dir, images_list, csv_name, num_classes, size=64):
    list_img = []
    for image, label in images_list:
        one_hot_label = np_utils.to_categorical([label], num_classes)
        list_img.append((image, label))
    list_img = np.array(list_img)

    hdf5_file = os.path.join(data_dir, "%s_%s_data.h5" % (csv_name, size))
    with h5py.File(hdf5_file, "w") as hfw:
        num_files = len(images_list)

        data_img = hfw.create_dataset("%s_image_data" % csv_name,
                                        (0, size, size, 3),
                                        maxshape=(None, size, size, 3),
                                        dtype=np.uint8)
        data_label = hfw.create_dataset("%s_label_data" % csv_name,
                                      (0, 1),
                                      maxshape=(None, num_classes),
                                      dtype=np.uint8)
        chunk_size = 1000
        num_chunks = num_files / chunk_size
        arr_chunks = np.array_split(np.arange(num_files), num_chunks)

        for chunk_idx in tqdm(arr_chunks):
            list_img_path = list_img[chunk_idx].tolist()
            output = parmap.starmap(format_data, list_img_path, size, parallel=True)

            arr_img = np.vstack([o[0] for o in output if o[0].shape[0] > 0])
            arr_label = np.vstack([o[1] for o in output if o[0].shape[0] > 0])

            data_img.resize(data_img.shape[0] + arr_img.shape[0], axis=0)
            data_label.resize(data_label.shape[0] + arr_label.shape[0], axis=0)

            data_img[-arr_img.shape[0]:] = arr_img.astype(np.uint8)
            data_label[-arr_label.shape[0]:] = arr_label.astype(np.uint8)


def check_HDF5(data_dir, csv_name, size=64):
    """
    Plot images with landmarks to check the processing
    """
    # Get hdf5 file
    hdf5_file = os.path.join(data_dir, "%s_%s_data.h5" % (csv_name, size))

    with h5py.File(hdf5_file, "r") as hf:
        data_color = hf["%s_image_data" % csv_name]
        data_label = hf["%s_label_data" % csv_name]
        for i in range(data_color.shape[0]):
            fig = plt.figure()
            gs = gridspec.GridSpec(1, 1)
            ax = plt.subplot(gs[0])
            img = data_color[i, :, :, :].transpose(1,2,0)
            ax.imshow(img)
            gs.tight_layout(fig)
            plt.title('Class %d' % (data_label[i]))
            plt.show()
            plt.clf()
            plt.close()


def load_image_pathes(csv_file_path):
    images_list = []
    with open(csv_file_path, 'r') as f:
        for line in f:
            file_label = line.split(",")
            images_list.append((file_label[0], int(file_label[1])))
    return images_list


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('--img_size', default=299, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--csv_file', default="train.csv", type=str,
                        help='transfer csv file path.')
    parser.add_argument('--data_dir', default="data", type=str,
                        help='data dir path.')
    parser.add_argument('--num_classes', default=25, type=int,
                        help='the number of classes.')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    size = args.img_size
    csv_file = args.csv_file
    images_list = load_image_pathes(csv_file)
    num_classes = args.num_classes

    csv_name, csv_ext = os.path.splitext(os.path.basename(csv_file))
    data_to_hdf5(args.data_dir, images_list, csv_name, num_classes, size=size)
    #check_HDF5(args.data_dir, csv_name)




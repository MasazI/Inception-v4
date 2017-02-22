# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import cv2
import os
import numpy as np
import argparse
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from inception_v4 import create_inception_v4
import matplotlib.pylab as plt
from PIL import Image

IMAGE_HOME = "clf_test"

cls_dict = {
    "bread_sandwich":  0,
    "bread_sliced":    1,
    "bread_sweets":    2,
    "bread_table":     3,
    "noodle_somen":    4,
    "noodle_udon":     5,
    "pasta_cream":     6,
    "pasta_gratin":    7,
    "pasta_japanese":  8,
    "pasta_oil":       9,
    "pasta_tomato":    10,
    "rice_boiled":     11,
    "rice_bowl":       12,
    "rice_curry":      13,
    "rice_fried":      14,
    "rice_risotto":    15,
    "rice_sushi":      16,
    "soup_miso":       17,
    "soup_potage":     18,
    "sweets_cheese":   19,
    "sweets_cookie":   20,
    "sweets_muffin":   21,
    "sweets_pie":      22,
    "sweets_pound":    23,
    "sweets_pudding":  24,
}


def format_data_clahe(img_path, size, limit=2):
    img_color = cv2.imread(img_path)
    img_color = img_color[:, :, ::-1]
    img_color = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_AREA)

    # clahe
    lab = cv2.cvtColor(img_color, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img_color = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_color = img_color.reshape((1, size, size, 3))
    return img_color


def format_data(img_path, size):
    """
    Load img with opencv and reshape
    """
    img_color = cv2.imread(img_path)
    img_color = img_color[:, :, ::-1]
    img_color = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_AREA)
    img_color = img_color.reshape((1, size, size, 3))\
    #.transpose(0, 3, 1, 2)

    return img_color


def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]


def format_data_10_crop(img_path, size, preprocess=True, verbose=False):
    img_color = cv2.imread(img_path)
    img_color = img_color[:, :, ::-1]
    # TODO superresolution
    img = cv2.resize(img_color, (500, 500), interpolation=cv2.INTER_AREA)
    flipped_X = np.fliplr(img)
    crops = [
        img[:size, :size, :],  # Upper Left
        img[:size, img.shape[1] - size:, :],  # Upper Right
        img[img.shape[0] - size:, :size, :],  # Lower Left
        img[img.shape[0] - size:, img.shape[1] - size:, :],  # Lower Right
        center_crop(img, (size, size)),

        flipped_X[:size, :size, :],
        flipped_X[:size, flipped_X.shape[1] - size:, :],
        flipped_X[flipped_X.shape[0] - size:, :size, :],
        flipped_X[flipped_X.shape[0] - size:, flipped_X.shape[1] - size:, :],
        center_crop(flipped_X, (size, size))
    ]

    if preprocess:
        pass

    if verbose:
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        ax[0][0].imshow(crops[0])
        ax[0][1].imshow(crops[1])
        ax[0][2].imshow(crops[2])
        ax[0][3].imshow(crops[3])
        ax[0][4].imshow(crops[4])
        ax[1][0].imshow(crops[5])
        ax[1][1].imshow(crops[6])
        ax[1][2].imshow(crops[7])
        ax[1][3].imshow(crops[8])
        ax[1][4].imshow(crops[9])
        plt.show()

    return np.array(crops)


def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def most_count_class(predict_clses):
    counter = np.zeros(25)
    for predict_cls in predict_clses:
        counter[predict_cls] += 1
    most_count_index = np.argmax(counter)
    return most_count_index


def predict_csv(**kwargs):
    csv_path = kwargs["file_path"]
    num_classes = kwargs["num_classes"]
    model_path = kwargs["model_path"]
    size = kwargs["size"]
    crop = kwargs["crop"]
    print("size: %d" % size)
    print("crop: %s" % crop)

    KTF.set_session(get_session())

    model = create_inception_v4(nb_classes=num_classes, load_weights=False, crop=crop)
    model.load_weights(model_path)

    model_name, ext = os.path.splitext(os.path.basename(model_path))

    with open(csv_path, 'r') as f:
        with open('result_top10_%s.csv' % model_name, 'w') as rf:
            with open('result_top10_for_app_%s.csv' % model_name, 'w') as raf:
                for i, img in enumerate(f):
                    img_path = 'clf_test/test_%d.jpg' % (i)
                    image_obj = format_data_10_crop(img_path, size, verbose=False)
                    preds = model.predict(image_obj)

                    predict_clses = []
                    predict_names = []
                    for pred in preds:
                        predict_cls = np.argmax(pred)
                        #top_n_preds = np.argpartition(pred, -5)[:, -5:]
                        predict_clses.append(predict_cls)
                        predict_name = ""
                        for name, age in cls_dict.iteritems():
                            if age == predict_cls:
                                predict_name = name
                                predict_names.append(predict_name)
                                break
                    predict_cls_10crop = most_count_class(predict_clses)
                    for name, age in cls_dict.iteritems():
                        if age == predict_cls_10crop:
                            predict_name_10crop = name
                            break
                    result = "test_%d.jpg,%d,%s" % (i, predict_cls_10crop, predict_name_10crop)
                    print(result)
                    rf.write(result)
                    rf.write("\n")
                    app_result = "%d,%d" % (i, predict_cls_10crop)
                    raf.write(app_result)
                    raf.write("\n")


def predict(**kwargs):
    file_path = kwargs["file_path"]
    num_classes = kwargs["num_classes"]
    model_path = kwargs["model_path"]

    model = create_inception_v4(nb_classes=num_classes, load_weights=False)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description='Inception-v4')
    parser.add_argument('mode', type=str, help='Choose csv or ind.')
    parser.add_argument('file_path', default="test.csv", type=str, help='file path for eval.')
    parser.add_argument('size', default=299, type=int, help='input rectangle size of input.')
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.set_defaults(crop=False)
    parser.add_argument('--num_classes', default=10, type=int, help='the number of classes.')
    parser.add_argument('--model_path', default="models/inception_v4/inception_v4_weights_epoch200.h5", type=str, help='model weights path.')
    args = parser.parse_args()

    d_params = {
                "file_path": args.file_path,
                "num_classes": args.num_classes,
                "model_path": args.model_path,
                "size": args.size,
                "crop": args.crop,
    }

    if args.mode == "csv":
        predict_csv(**d_params)
    elif args.mode == "inv":
        predict(**d_params)



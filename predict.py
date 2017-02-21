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


def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
        with open('result_%s.csv' % model_name, 'w') as rf:
            with open('result_fro_app_%s.csv' % model_name, 'w') as raf:
                for i, img in enumerate(f):
                    img_path = 'clf_test/test_%d.jpg' % (i)
                    image_obj = format_data(img_path, size)
                    preds = model.predict(image_obj)
                    predict_cls = np.argmax(preds)
                    predict_name = ""
                    for name, age in cls_dict.iteritems():
                        if age == predict_cls:
                            predict_name = name
                            break
                    result = "test_%d.jpg,%d,%s" % (i, predict_cls, predict_name)
                    print(result)
                    rf.write(result)
                    rf.write("\n")
                    app_result = "%d,%d" % (i, predict_cls)
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



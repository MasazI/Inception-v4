# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import cv2
import os
import numpy as np
import argparse
from inception_v4 import create_inception_v4

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


def predict_csv(**kwargs):
    csv_path = kwargs["file_path"]
    num_classes = kwargs["num_classes"]
    model_path = kwargs["model_path"]

    model = create_inception_v4(nb_classes=num_classes, load_weights=False)
    model.load_weights(model_path)

    model_name, ext = os.path.splitext(os.path.basename(model_path))

    with open(csv_path, 'r') as f:
        with open('result_%s.csv' % model_name, 'w') as rf:
            with open('result_fro_app_%s.csv' % model_name, 'w') as raf:
                for i, img in enumerate(f):
                    img_path = 'clf_test/test_%d.jpg' % (i)
                    image_obj = format_data(img_path, 299)
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
    parser.add_argument('--num_classes', default=10, type=int, help='the number of classes.')
    parser.add_argument('--model_path', default="models/inception_v4/inception_v4_weights_epoch395.h5", type=str, help='model weights path.')
    args = parser.parse_args()

    d_params = {
                "file_path": args.file_path,
                "num_classes": args.num_classes,
                "model_path": args.model_path
                }

    if args.mode == "csv":
        predict_csv(**d_params)
    elif args.mode == "inv":
        predict(**d_params)



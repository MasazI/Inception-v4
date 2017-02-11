from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import time
import argparse
import batches
from keras.optimizers import Adam
import keras.preprocessing.image as keras_image
from inception_v4 import create_inception_v4

DEBUG = False

def eval():
    pass

def train(**kwargs):
    data_file = kwargs["data_file"]
    data_dir = kwargs["data_dir"]
    model_name = kwargs["model_name"]
    batch_size = kwargs["batch_size"]
    nb_epoch = kwargs["nb_epoch"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    num_classes = kwargs["num_classes"]
    epoch_size = n_batch_per_epoch * batch_size

    img_size = int(os.path.basename(data_file).split("_")[1])

    # Create a batch generator for the color data
    data_path = os.path.join(data_dir, data_file)
    DataGen = batches.DataGenerator(data_path,
                                        num_classes=num_classes,
                                        batch_size=batch_size,
                                        dset='train')
    h, w, c = DataGen.get_config()["data_shape"][1:]

    print("input data info")
    print("- channel: %d" % c)
    print("- height: %d" % h)
    print("- width: %d" % w)

    try:

        if model_name == 'inception_v4':
            model = create_inception_v4(nb_classes=num_classes, load_weights=True)
        elif model_name == 'inception_resnet_v1':
            pass
        elif model_name == 'inception_resnet_v2':
            pass

        model.summary()

        opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model.compile(optimizer=opt, loss='categorical_crossentropy')

        model_dir = os.path.join("models", model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for epoch in range(nb_epoch):
            start = time.time()
            batch_counter = 1
            BatchGen = DataGen.gen_batch()
            for batch in BatchGen:
                X_batch, Y_batch = batch
                if DEBUG:
                    print("X batch shape: %s" % (X_batch.shape,))

                X_batch_list = []
                for i, x in enumerate(X_batch):
                    if DEBUG:
                        print("batch shape: %s" % (x.shape,))
                    # rotation
                    x = keras_image.random_rotation(x, rg=40, row_axis=2, col_axis=1, channel_axis=2)
                    # shift
                    x = keras_image.random_shift(x, 0.1, 0.1, row_axis=2, col_axis=1, channel_axis=2)
                    # share
                    x = keras_image.random_shear(x, 0.2, row_axis=2, col_axis=1, channel_axis=2)
                    # zoom
                    x = keras_image.random_zoom(x, (0.7, 1.3), row_axis=2, col_axis=1, channel_axis=2)
                    # flip horizontal
                    if np.random.random() < 0.5:
                        x = keras_image.flip_axis(x, axis=1)

                    if DEBUG:
                        x_disp = x[:,:,::-1]
                        cv2.imshow("image", x_disp)
                        cv2.waitKey(0)
                        print("batch shape: %s" % (x.shape,))
                    X_batch_list.append(x)
                arr_img = np.array(X_batch_list)

                if DEBUG:
                    print("X batch aug shape: %s" % (arr_img.shape,))
                    for sample in arr_img:
                        sample_disp = sample[:,:,::-1]
                        cv2.imshow("image", sample_disp)
                        cv2.waitKey(0)

                train_loss = model.train_on_batch(arr_img, Y_batch)
                batch_counter += 1
                if batch_counter >= n_batch_per_epoch:
                    print('Train loss: %f' % train_loss)
                    break
            print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start))

            if epoch % 20 == 0:
                weights_path = os.path.join('models/%s/%s_weights_epoch%s.h5' %
                                            (model_name, model_name, epoch))
                model.save_weights(weights_path, overwrite=True)
    except Exception as e:
        print(e)
        pass




if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='Inception-v4')

    parser.add_argument('mode', type=str, help="Choose train or eval")
    parser.add_argument('data_file', type=str, help="Path to HDF5 containing the data")
    parser.add_argument('--data_dir', default="data", type=str,
                        help='data dir path.')
    parser.add_argument('--model_name', type=str, default="inception_v4",
                        help="Model name. Choose inception_v4, inception_resnet_v1 or inception_resnet_v2")
    # parser.add_argument('--training_mode', default="in_memory", type=str,
    #                     help=('Training mode. Choose in_memory to load all the data in memory and train.'
    #                           'Choose on_demand to load batches from disk at each step'))
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--epoch', default=0, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--num_classes', default=10, type=int, help='the number of classes.')

    args = parser.parse_args()

    # Set default params
    d_params = {"data_file": args.data_file,
                "data_dir": args.data_dir,
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "nb_epoch": args.nb_epoch,
                # "training_mode": args.training_mode,
                "epoch": args.epoch,
                "num_classes": args.num_classes
                }

    if args.mode == 'train':
        data_file = args.data_file
        data_dir = args.data_dir
        train(**d_params)
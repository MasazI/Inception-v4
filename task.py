from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import batches
from keras.optimizers import Adam
from inception_v4 import create_inception_v4

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

        # try:
        #     model.load_weights(filepath=os.path.join('weights', 'inception-v4_weights_tf_dim_ordering_tf_kernels.h5'))
        # except Exception as e:
        #     print("[Error] %s" % e)

        model_dir = os.path.join("models", model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for epoch in range(nb_epoch):
            start = time.time()
            batch_counter = 1
            BatchGen = DataGen.gen_batch()
            for batch in BatchGen:
                X_batch, Y_batch = batch
                train_loss = model.train_on_batch(X_batch, Y_batch)
                batch_counter += 1
                if batch_counter >= n_batch_per_epoch:
                    print('Train loss: %f' % train_loss)
                    break
            print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start))

            if epoch % 5 == 0:
                weights_path = os.path.join('models/%s/%s_weights_epoch%s.h5' %
                                            (model_name, model_name, epoch))
                model.save_weights(weights_path, overwrite=True)
    except:
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
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
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
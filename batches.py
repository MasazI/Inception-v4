from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import multiprocessing
import h5py
from keras.utils import np_utils
import cv2


def clahe(x, size, limit=2):
    # clahe
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img_color = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    #img_color = img_color.reshape((1, size, size, 3))
    return img_color


class DataGenerator(object):
    """
    Generate minibatches with real-time data parallel augmentation on CPU
    args :
        hdf5_file   (str)      path to data in HDF5 format
        batch_size  (int)      Minibatch size
        dset        (str)      for examples train/test/valid, the name of the dset to iterate over
        maxproc     (int)      max number of processes to spawn in parallel
        num_cached  (int)      max number of batches to keep in queue
    yields :
         X, y (minibatch data and labels as np arrays)
    """

    def __init__(self,
                 hdf5_file,
                 num_classes=10,
                 batch_size=32,
                 dset="test",
                 maxproc=8,
                 num_cached=10):

        # Check file exists
        assert os.path.isfile(hdf5_file), hdf5_file + " doesn't exist"

        # Initialize class internal variables
        self.dset = dset
        self.maxproc = maxproc
        self.hdf5_file = hdf5_file
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_cached = num_cached

        # Dict that will store all transformations and their parameters
        self.d_transform = {}

        # Read the data file to get dataset shape information
        with h5py.File(self.hdf5_file, "r") as hf:
            self.X_shape = hf["%s_image_data" % self.dset].shape
            assert len(self.X_shape) == 4, \
                ("\n\nImg data should be formatted as(1): \n"
                 "(n_samples, Height, Width, n_channels)")

            self.n_samples = hf["%s_label_data" % self.dset].shape[0]
            # Verify n_channels is at index 1
            assert self.X_shape[-3] > min(self.X_shape[-2:]), \
                ("\n\nImg data should be formatted as(2): \n"
                 "(n_samples, Height, Width, n_channels)")

        # Save the class internal variables to a config dict
        self.d_config = {}
        self.d_config["hdf5_file"] = hdf5_file
        self.d_config["batch_size"] = batch_size
        self.d_config["dset"] = dset
        self.d_config["num_cached"] = num_cached
        self.d_config["maxproc"] = maxproc
        self.d_config["data_shape"] = self.X_shape

    def get_config(self):
        return self.d_config

    def gen_batch(self):
        """ Use multiprocessing to generate batches in parallel. """
        try:
            queue = multiprocessing.Queue(maxsize=self.num_cached)

            # define producer (putting items into queue)
            def producer():

                try:
                    # Load the data from HDF5 file
                    with h5py.File(self.hdf5_file, "r") as hf:
                        num_chan, height, width = self.X_shape[-3:]
                        # Select start_idx at random for the batch
                        idx_start = np.random.randint(0, self.X_shape[0] - self.batch_size)
                        idx_end = idx_start + self.batch_size
                        # Get X and y
                        X_batch_color = hf["%s_image_data" % self.dset][idx_start: idx_end, :, :, :]
                        Y_batch = hf["%s_label_data" % self.dset][idx_start: idx_end]
                        # to one hot vector.
                        Y_batch = np_utils.to_categorical(Y_batch, self.num_classes)
                        # Put the data in a queue
                        queue.put((X_batch_color, Y_batch))
                except Exception as e:
                    print(e)
                    print("Nothing here")

            processes = []

            def start_process():
                for i in range(len(processes), self.maxproc):
                    # Reset the seed ! (else the processes share the same seed)
                    np.random.seed()
                    thread = multiprocessing.Process(target=producer)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            while True:
                processes = [p for p in processes if p.is_alive()]

                if len(processes) < self.maxproc:
                    start_process()

                yield queue.get()
        except:
            for th in processes:
                th.terminate()
            queue.close()
            raise


if __name__ == '__main__':
    hdf5_file = os.path.join("data", "%s_%s_data.h5" % ("test", 64))
    dg = DataGenerator(hdf5_file, batch_size=2)
    print(dg.get_config())
    batches = dg.gen_batch()
    for batch in batches:
        print(batch)
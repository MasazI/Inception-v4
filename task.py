import tensorflow as tf
from inception_v4 import create_inception_v4


def train():
    model = create_inception_v4()


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
import tensorflow as tf
from tensorflow import keras


class Callback(keras.callbacks.Callback):
    """
    自定义回调，暂无作用
    """
    def on_epoch_end(self, epoch, logs=None):
        # print(epoch, logs)
        pass

    def on_batch_end(self, batch, logs=None):
        # print('\n', batch, logs)
        pass

    pass

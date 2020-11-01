import tensorflow as tf


class Callback(tf.keras.callbacks.Callback):
    def __init__(self, accuracy: float = None, val_acc: float = None):
        self.accuracy = accuracy
        self.val_acc = val_acc

    def on_epoch_end(self, epoch, logs=None):
        if self.accuracy and logs.get('accuracy') > self.accuracy:
            self.model.stop_training = True
            print(f'\nReached {self.accuracy * 100}% training accuracy so cancelling training!')
        if self.val_acc and logs.get('val_accuracy') > self.val_acc:
            self.model.stop_training = True
            print(f'\nReached {self.val_acc * 100}% validation accuracy so cancelling training!')

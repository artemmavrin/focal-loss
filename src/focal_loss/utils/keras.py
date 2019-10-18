"""Keras-related utilities."""

import tensorflow as tf


def register_keras_custom_object(cls):
    """Register a class as a custom Keras object.

    Taken from
    https://github.com/tensorflow/addons/blob/40de2b942c833c6afccd96759f106640b983953b/tensorflow_addons/utils/keras_utils.py#L23

    Parameters
    ----------
    cls : type
        A class.

    Returns
    -------
    type
        The class `cls`.
    """
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls

"""Focal loss unit tests."""

from itertools import product
from math import exp
import os
import shutil

import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid

from focal_loss import binary_focal_loss, BinaryFocalLoss

# Synthetic label/prediction data as pure Python lists
Y_TRUE_LIST = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Y_PRED_LOGITS_LIST = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
Y_PRED_PROB_LIST = [1 / (1 + exp(-y)) for y in Y_PRED_LOGITS_LIST]

# Synthetic label/prediction data as NumPy arrays
Y_TRUE_ARRAY = np.asarray(Y_TRUE_LIST, dtype=np.int64)
Y_PRED_LOGITS_ARRAY = np.asarray(Y_PRED_LOGITS_LIST, dtype=np.float32)
Y_PRED_PROB_ARRAY = sigmoid(Y_PRED_LOGITS_ARRAY)

# Synthetic label/prediction data as TensorFlow tensors
Y_TRUE_TENSOR = tf.convert_to_tensor(Y_TRUE_LIST, dtype=tf.int64)
Y_PRED_LOGITS_TENSOR = tf.convert_to_tensor(Y_PRED_LOGITS_LIST,
                                            dtype=tf.float32)
Y_PRED_PROB_TENSOR = tf.math.sigmoid(Y_PRED_LOGITS_TENSOR)

Y_TRUE = [Y_TRUE_LIST, Y_TRUE_ARRAY, Y_TRUE_TENSOR]
Y_PRED_LOGITS = [Y_PRED_LOGITS_LIST, Y_PRED_LOGITS_ARRAY, Y_PRED_LOGITS_TENSOR]
Y_PRED_PROB = [Y_PRED_PROB_LIST, Y_PRED_PROB_ARRAY, Y_PRED_PROB_TENSOR]


def numpy_binary_focal_loss(y_true, y_pred, gamma, from_logits=False,
                            pos_weight=None, label_smoothing=None):
    """Simple focal loss implementation using NumPy."""
    # Convert to arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if from_logits:
        y_pred = sigmoid(y_pred)

    # Apply label smoothing
    if label_smoothing is not None:
        y_true = y_true.astype(y_pred.dtype)
        y_true = (1 - label_smoothing) * y_true + label_smoothing * 0.5

    loss = -y_true * ((1 - y_pred) ** gamma) * np.log(y_pred)
    if pos_weight is not None:
        loss *= pos_weight
    loss -= (1 - y_true) * (y_pred ** gamma) * np.log(1 - y_pred)
    return loss


def test_computation_sanity_checks():
    """Make sure the focal loss computation behaves as expected."""
    for y_true, y_pred_logits in product(Y_TRUE, Y_PRED_LOGITS):
        for y_pred_prob, pos_weight in product((Y_PRED_PROB), (None, 1, 2)):
            for gamma, label_smoothing in product((0, 1, 2), (None, 0.1, 0.5)):
                focal_loss_prob = binary_focal_loss(
                    y_true=y_true,
                    y_pred=y_pred_prob,
                    gamma=gamma,
                    from_logits=False,
                    pos_weight=pos_weight,
                    label_smoothing=label_smoothing,
                )
                focal_loss_logits = binary_focal_loss(
                    y_true=y_true,
                    y_pred=y_pred_logits,
                    gamma=gamma,
                    from_logits=True,
                    pos_weight=pos_weight,
                    label_smoothing=label_smoothing,
                )
                losses = [focal_loss_prob, focal_loss_logits]
                if not (isinstance(y_true, tf.Tensor)
                        or isinstance(y_pred_logits, tf.Tensor)):
                    numpy_focal_loss_logits = numpy_binary_focal_loss(
                        y_true=y_true,
                        y_pred=y_pred_logits,
                        gamma=gamma,
                        from_logits=True,
                        pos_weight=pos_weight,
                        label_smoothing=label_smoothing,
                    )
                    losses.append(numpy_focal_loss_logits)
                if not (isinstance(y_true, tf.Tensor)
                        or isinstance(y_pred_prob, tf.Tensor)):
                    numpy_focal_loss_prob = numpy_binary_focal_loss(
                        y_true=y_true,
                        y_pred=y_pred_prob,
                        gamma=gamma,
                        from_logits=False,
                        pos_weight=pos_weight,
                        label_smoothing=label_smoothing,
                    )
                    losses.append(numpy_focal_loss_prob)

                for i, loss_1 in enumerate(losses):
                    for loss_2 in losses[(i + 1):]:
                        tf.debugging.assert_near(loss_1, loss_2, atol=1e-5,
                                                 rtol=1e-5)


def test_reduce_to_binary_crossentropy():
    """Focal loss with gamma=0 should be the same as cross-entropy."""
    # From probabilities
    for y_true, y_pred in product(Y_TRUE, Y_PRED_PROB):
        # tf.keras.losses.binary_crossentropy averages its output along the last
        # axis, so we do the same here
        focal_loss = binary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0)
        focal_loss = tf.math.reduce_mean(focal_loss, axis=-1)
        ce = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
        tf.debugging.assert_near(focal_loss, ce)

    # From logits
    for y_true, y_pred in product(Y_TRUE, Y_PRED_LOGITS):
        focal_loss = binary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0,
                                       from_logits=True)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.dtypes.cast(y_true, dtype=tf.float32),
            logits=tf.dtypes.cast(y_pred, dtype=tf.float32),
        )
        tf.debugging.assert_near(focal_loss, ce)

    # From logits, with positive class weighting
    for y_true, y_pred, pos_weight in product(Y_TRUE, Y_PRED_LOGITS, (1, 2)):
        focal_loss = binary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0,
                                       from_logits=True, pos_weight=pos_weight)
        ce = tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.dtypes.cast(y_true, dtype=tf.float32),
            logits=tf.dtypes.cast(y_pred, dtype=tf.float32),
            pos_weight=pos_weight,
        )
        tf.debugging.assert_near(focal_loss, ce)


def test_train_dummy_binary_classifier():
    """Train a simple model to make sure that BinaryFocalLoss works."""
    # Data/model parameters
    n_examples = 100
    n_features = 16
    epochs = 3
    random_state = np.random.RandomState(0)

    # Generate some fake data
    x = random_state.binomial(n=1, p=0.5, size=(n_examples, n_features))
    x = 2.0 * x.astype(np.float32) - 1.0
    weights = 100 * np.ones(shape=(n_features, 1)).astype(np.float32)
    y = (x.dot(weights) > 0).astype(np.int8)

    # Number of positive and negative examples
    n_pos = y.sum()
    n_neg = n_examples - n_pos

    for pos_weight in (None, (n_neg / n_pos)):
        for gamma, label_smoothing in product((0, 2), (None, 0.1)):
            for from_logits in (True, False):
                if from_logits:
                    activation = None
                else:
                    activation = 'sigmoid'
                # Just a linear classifier (without bias term)
                model = tf.keras.Sequential(layers=[
                    tf.keras.layers.Input(shape=n_features),
                    tf.keras.layers.Dense(units=1, use_bias=False,
                                          activation=activation),
                ])
                model.compile(
                    optimizer='sgd',
                    loss=BinaryFocalLoss(gamma=gamma, pos_weight=pos_weight,
                                         from_logits=from_logits,
                                         label_smoothing=label_smoothing),
                    metrics=['accuracy'],
                )
                stop_on_nan = tf.keras.callbacks.TerminateOnNaN()
                history = model.fit(x, y, batch_size=n_examples, epochs=epochs,
                                    callbacks=[stop_on_nan])
                history = history.history

                # Check that we didn't stop early: if we did then we
                # encountered NaNs during training, and that shouldn't happen
                assert len(history['loss']) == epochs

                # Check that BinaryFocalLoss and binary_focal_loss agree (at
                # least when averaged)
                model_loss, *_ = model.evaluate(x, y)

                y_pred = model.predict(x)
                loss = binary_focal_loss(y_true=y, y_pred=y_pred,
                                         gamma=gamma, pos_weight=pos_weight,
                                         from_logits=from_logits,
                                         label_smoothing=label_smoothing)
                loss = tf.math.reduce_mean(loss)
                tf.debugging.assert_near(loss, model_loss)


def test_get_config():
    """Check the get_config() method."""
    for gamma, pos_weight in product((0, 1, 2), (None, 0.5)):
        for from_logits, label_smoothing in product((True, False), (None, 0.1)):
            loss1 = BinaryFocalLoss(gamma=gamma, pos_weight=pos_weight,
                                    from_logits=from_logits,
                                    label_smoothing=label_smoothing,
                                    name='binary_focal_loss')
            config1 = loss1.get_config()
            loss2 = BinaryFocalLoss(**config1)
            config2 = loss2.get_config()
            assert config1 == config2


def test_save_and_restore():
    """Check whether models compiled with BinaryFocalLoss can be saved/loaded.
    """
    n_features = 10
    for gamma, pos_weight in product((0, 1, 2), (None, 0.5)):
        for from_logits, label_smoothing in product((True, False), (None, 0.1)):
            if from_logits:
                activation = None
            else:
                activation = 'sigmoid'
            # Just a linear classifier
            model = tf.keras.Sequential(layers=[
                tf.keras.layers.Input(shape=n_features),
                tf.keras.layers.Dense(units=1, activation=activation),
            ])
            model.compile(
                optimizer='sgd',
                loss=BinaryFocalLoss(gamma=gamma, pos_weight=pos_weight,
                                     from_logits=from_logits,
                                     label_smoothing=label_smoothing),
                metrics=['accuracy'],
            )
            weights = model.weights

            # Try to save the model to the HDF5 format
            h5_filepath = 'model.h5'
            model.save(h5_filepath, save_format='h5')

            h5_restored_model = tf.keras.models.load_model(h5_filepath)
            h5_restored_weights = h5_restored_model.weights
            for weight, h5_restored_weight in zip(weights, h5_restored_weights):
                tf.debugging.assert_equal(weight, h5_restored_weight)

            # Delete the created HDF5 file
            os.unlink(h5_filepath)

            # Try to save the model to the SavedModel format
            sm_filepath = 'model'
            model.save(sm_filepath, save_format='tf')

            sm_restored_model = tf.keras.models.load_model(sm_filepath)
            sm_restored_weights = sm_restored_model.weights
            for weight, sm_restored_weight in zip(weights, sm_restored_weights):
                tf.debugging.assert_equal(weight, sm_restored_weight)

            # Delete the created SavedModel directory
            shutil.rmtree(sm_filepath, ignore_errors=True)

"""Binary focal loss unit tests."""

from math import exp
import os
import shutil
from typing import Optional

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid

from focal_loss import binary_focal_loss, BinaryFocalLoss
from .utils import named_parameters_with_testcase_names

# Synthetic label/prediction data as pure Python lists
Y_TRUE_LIST = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Y_PRED_LOGITS_LIST = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
Y_PRED_PROB_LIST = [1 / (1 + exp(-y)) for y in Y_PRED_LOGITS_LIST]

# Synthetic label/prediction data as NumPy arrays
Y_TRUE_ARRAY = np.asarray(Y_TRUE_LIST, dtype=np.int64)
Y_PRED_LOGITS_ARRAY = np.asarray(Y_PRED_LOGITS_LIST, dtype=np.float32)
Y_PRED_PROB_ARRAY = sigmoid(Y_PRED_LOGITS_ARRAY)

# Synthetic label/prediction data as TensorFlow tensors
Y_TRUE_TENSOR = tf.convert_to_tensor(Y_TRUE_LIST, dtype=tf.dtypes.int64)
Y_PRED_LOGITS_TENSOR = tf.convert_to_tensor(Y_PRED_LOGITS_LIST,
                                            dtype=tf.dtypes.float32)
Y_PRED_PROB_TENSOR = tf.math.sigmoid(Y_PRED_LOGITS_TENSOR)

Y_TRUE = [Y_TRUE_LIST, Y_TRUE_ARRAY, Y_TRUE_TENSOR]
Y_PRED_LOGITS = [Y_PRED_LOGITS_LIST, Y_PRED_LOGITS_ARRAY, Y_PRED_LOGITS_TENSOR]
Y_PRED_PROB = [Y_PRED_PROB_LIST, Y_PRED_PROB_ARRAY, Y_PRED_PROB_TENSOR]


def numpy_binary_focal_loss(y_true, y_pred, gamma, from_logits=False,
                            pos_weight=None, label_smoothing=None):
    """Simple binary focal loss implementation using NumPy."""
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


def get_dummy_binary_classifier(n_features, gamma, pos_weight,
                                label_smoothing, from_logits):
    activation = None if from_logits else 'sigmoid'

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

    return model


class BinaryFocalLossTest(parameterized.TestCase, tf.test.TestCase):
    @named_parameters_with_testcase_names(
        y_true=Y_TRUE, y_pred_logits=Y_PRED_LOGITS, y_pred_prob=Y_PRED_PROB,
        pos_weight=[None, 1, 2], gamma=[0, 1, 2],
        label_smoothing=[None, 0.1, 0.5])
    def test_computation_sanity_checks(self, y_true, y_pred_logits, y_pred_prob,
                                       pos_weight, gamma, label_smoothing):
        """Make sure the focal loss computation behaves as expected."""
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
                self.assertAllClose(loss_1, loss_2, atol=1e-5, rtol=1e-5)

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_PROB)
    def test_reduce_to_binary_crossentropy_from_probabilities(self, y_true,
                                                              y_pred):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        # tf.keras.losses.binary_crossentropy averages its output along the last
        # axis, so we do the same here
        focal_loss = binary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0)
        focal_loss = tf.math.reduce_mean(focal_loss, axis=-1)
        ce = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
        self.assertAllClose(focal_loss, ce)

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_LOGITS)
    def test_reduce_to_binary_crossentropy_from_logits(self, y_true, y_pred):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        focal_loss = binary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0,
                                       from_logits=True)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.dtypes.cast(y_true, dtype=tf.dtypes.float32),
            logits=tf.dtypes.cast(y_pred, dtype=tf.dtypes.float32),
        )
        self.assertAllClose(focal_loss, ce)

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_LOGITS,
                                          pos_weight=[1, 2])
    def test_reduce_to_binary_crossentropy_with_weighting(self, y_true, y_pred,
                                                          pos_weight):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        focal_loss = binary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0,
                                       from_logits=True, pos_weight=pos_weight)
        ce = tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.dtypes.cast(y_true, dtype=tf.dtypes.float32),
            logits=tf.dtypes.cast(y_pred, dtype=tf.dtypes.float32),
            pos_weight=pos_weight,
        )
        self.assertAllClose(focal_loss, ce)

    def _test_reduce_to_keras_loss(self, y_true, y_pred, from_logits: bool,
                                   label_smoothing: Optional[float]):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.dtypes.float32)
        keras_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=(0 if label_smoothing is None else label_smoothing),
        )
        focal_loss = BinaryFocalLoss(
            gamma=0, from_logits=from_logits, label_smoothing=label_smoothing)
        self.assertAllClose(keras_loss(y_true, y_pred),
                            focal_loss(y_true, y_pred))

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_LOGITS,
                                          label_smoothing=[None, 0.1, 0.3])
    def test_reduce_to_keras_loss_logits(self, y_true, y_pred, label_smoothing):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        self._test_reduce_to_keras_loss(y_true, y_pred, from_logits=True,
                                        label_smoothing=label_smoothing)

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_PROB,
                                          label_smoothing=[None, 0.1, 0.3])
    def test_reduce_to_keras_loss_probabilities(self, y_true, y_pred,
                                                label_smoothing):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        self._test_reduce_to_keras_loss(y_true, y_pred, from_logits=False,
                                        label_smoothing=label_smoothing)

    @named_parameters_with_testcase_names(
        n_examples=100, n_features=16, epochs=3, pos_weight=[None, 0.5],
        gamma=[0, 2], label_smoothing=[None, 0.1], from_logits=[True, False],
        random_state=np.random.default_rng(0))
    def test_train_dummy_binary_classifier(self, n_examples, n_features, epochs,
                                           pos_weight, gamma, label_smoothing,
                                           from_logits, random_state):
        # Generate some fake data
        x = random_state.binomial(n=1, p=0.5, size=(n_examples, n_features))
        x = 2.0 * x.astype(np.float32) - 1.0
        weights = 100 * np.ones(shape=(n_features, 1)).astype(np.float32)
        y = (x.dot(weights) > 0).astype(np.int8)

        model = get_dummy_binary_classifier(n_features=n_features, gamma=gamma,
                                            pos_weight=pos_weight,
                                            label_smoothing=label_smoothing,
                                            from_logits=from_logits)
        history = model.fit(x, y, batch_size=n_examples, epochs=epochs,
                            callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        history = history.history

        # Check that we didn't stop early: if we did then we
        # encountered NaNs during training, and that shouldn't happen
        self.assertEqual(len(history['loss']), epochs)

        # Check that BinaryFocalLoss and binary_focal_loss agree (at
        # least when averaged)
        model_loss, *_ = model.evaluate(x, y)

        y_pred = model.predict(x)
        loss = binary_focal_loss(y_true=y, y_pred=y_pred,
                                 gamma=gamma, pos_weight=pos_weight,
                                 from_logits=from_logits,
                                 label_smoothing=label_smoothing)
        loss = tf.math.reduce_mean(loss)
        self.assertAllClose(loss, model_loss)

    @named_parameters_with_testcase_names(
        gamma=[0, 1, 2], pos_weight=[None, 0.5], from_logits=[False, True],
        label_smoothing=[None, 0.1])
    def test_get_config(self, gamma, pos_weight, from_logits, label_smoothing):
        """Check the get_config() method."""
        loss1 = BinaryFocalLoss(gamma=gamma, pos_weight=pos_weight,
                                from_logits=from_logits,
                                label_smoothing=label_smoothing,
                                name='binary_focal_loss')
        config1 = loss1.get_config()
        loss2 = BinaryFocalLoss(**config1)
        config2 = loss2.get_config()
        self.assertEqual(config1, config2)

    @named_parameters_with_testcase_names(
        gamma=[0, 1, 2], pos_weight=[None, 0.5], from_logits=[False, True],
        label_smoothing=[None, 0.1])
    def test_save_and_restore(self, gamma, pos_weight, from_logits,
                              label_smoothing):
        """Check if models compiled with BinaryFocalLoss can be saved/loaded.
        """
        model = get_dummy_binary_classifier(n_features=10, gamma=gamma,
                                            pos_weight=pos_weight,
                                            label_smoothing=label_smoothing,
                                            from_logits=from_logits)
        weights = model.weights

        temp_dir = self.get_temp_dir()

        # Try to save the model to the HDF5 format
        h5_filepath = os.path.join(temp_dir, 'model.h5')
        model.save(h5_filepath, save_format='h5')

        h5_restored_model = tf.keras.models.load_model(h5_filepath)
        h5_restored_weights = h5_restored_model.weights
        for weight, h5_restored_weight in zip(weights, h5_restored_weights):
            self.assertAllClose(weight, h5_restored_weight)

        # Delete the created HDF5 file
        os.unlink(h5_filepath)

        # Try to save the model to the SavedModel format
        sm_filepath = os.path.join(temp_dir, 'model')
        model.save(sm_filepath, save_format='tf')

        sm_restored_model = tf.keras.models.load_model(sm_filepath)
        sm_restored_weights = sm_restored_model.weights
        for weight, sm_restored_weight in zip(weights, sm_restored_weights):
            self.assertAllClose(weight, sm_restored_weight)

        # Delete the created SavedModel directory
        shutil.rmtree(sm_filepath, ignore_errors=True)

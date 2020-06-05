"""Sparse categorical focal loss unit tests."""

from math import exp
import os
import shutil

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from scipy.special import softmax

from focal_loss import sparse_categorical_focal_loss, SparseCategoricalFocalLoss
from .utils import named_parameters_with_testcase_names

# Synthetic label/prediction data as pure Python lists
Y_TRUE_LIST = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
Y_PRED_LOGITS_LIST = [
    [6., 9., 2.],
    [7., 8., 1.],
    [9., 7., 9.],
    [3., 2., 9.],
    [3., 9., 4.],
    [0., 5., 7.],
    [2., 0., 3.],
    [4., 7., 4.],
    [6., 2., 4.],
    [8., 6., 9.],
    [0., 0., 3.],
    [6., 6., 4.],
    [3., 9., 5.],
    [7., 5., 3.],
    [4., 6., 0.]
]
Y_PRED_PROB_LIST = [
    [exp(y - max(row)) / sum(exp(z - max(row)) for z in row) for y in row]
    for row in Y_PRED_LOGITS_LIST
]

# Synthetic label/prediction data as NumPy arrays
Y_TRUE_ARRAY = np.asarray(Y_TRUE_LIST, dtype=np.int64)
Y_PRED_LOGITS_ARRAY = np.asarray(Y_PRED_LOGITS_LIST, dtype=np.float32)
Y_PRED_PROB_ARRAY = softmax(Y_PRED_LOGITS_ARRAY, axis=-1)

# Synthetic label/prediction data as TensorFlow tensors
Y_TRUE_TENSOR = tf.convert_to_tensor(Y_TRUE_LIST, dtype=tf.int64)
Y_PRED_LOGITS_TENSOR = tf.convert_to_tensor(Y_PRED_LOGITS_LIST,
                                            dtype=tf.float32)
Y_PRED_PROB_TENSOR = tf.nn.softmax(Y_PRED_LOGITS_TENSOR)

Y_TRUE = [Y_TRUE_LIST, Y_TRUE_ARRAY, Y_TRUE_TENSOR]
Y_PRED_LOGITS = [Y_PRED_LOGITS_LIST, Y_PRED_LOGITS_ARRAY, Y_PRED_LOGITS_TENSOR]
Y_PRED_PROB = [Y_PRED_PROB_LIST, Y_PRED_PROB_ARRAY, Y_PRED_PROB_TENSOR]


def numpy_sparse_categorical_focal_loss(y_true, y_pred, gamma,
                                        from_logits=False):
    """Simple sparse categorical focal loss implementation using NumPy."""
    # Convert to arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # One-hot encoding of integer labels
    y_true_one_hot = np.eye(y_pred.shape[-1])[y_true]

    if from_logits:
        y_pred = softmax(y_pred, axis=-1)

    loss = -y_true_one_hot * (1 - y_pred) ** gamma * np.log(y_pred)
    return loss.sum(axis=-1)


def get_dummy_sparse_multiclass_classifier(n_features, n_classes, gamma,
                                           from_logits):
    activation = None if from_logits else 'softmax'

    # Just a linear classifier (without bias term)
    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Input(shape=n_features),
        tf.keras.layers.Dense(units=n_classes, use_bias=False,
                              activation=activation),
    ])
    model.compile(
        optimizer='sgd',
        loss=SparseCategoricalFocalLoss(gamma=gamma, from_logits=from_logits),
        metrics=['accuracy'],
    )

    return model


class SparseCategoricalFocalLossTest(parameterized.TestCase, tf.test.TestCase):
    @named_parameters_with_testcase_names(
        y_true=Y_TRUE, y_pred_logits=Y_PRED_LOGITS, y_pred_prob=Y_PRED_PROB,
        gamma=[0, 1, 2, [2, 2, 2]])
    def test_computation_sanity_checks(self, y_true, y_pred_logits, y_pred_prob,
                                       gamma):
        """Make sure the focal loss computation behaves as expected."""
        focal_loss_prob = sparse_categorical_focal_loss(
            y_true=y_true,
            y_pred=y_pred_prob,
            gamma=gamma,
            from_logits=False,
        )
        focal_loss_logits = sparse_categorical_focal_loss(
            y_true=y_true,
            y_pred=y_pred_logits,
            gamma=gamma,
            from_logits=True,
        )
        losses = [focal_loss_prob, focal_loss_logits]
        if not (isinstance(y_true, tf.Tensor)
                or isinstance(y_pred_logits, tf.Tensor)):
            numpy_focal_loss_logits = numpy_sparse_categorical_focal_loss(
                y_true=y_true,
                y_pred=y_pred_logits,
                gamma=gamma,
                from_logits=True,
            )
            losses.append(numpy_focal_loss_logits)
        if not (isinstance(y_true, tf.Tensor)
                or isinstance(y_pred_prob, tf.Tensor)):
            numpy_focal_loss_prob = numpy_sparse_categorical_focal_loss(
                y_true=y_true,
                y_pred=y_pred_prob,
                gamma=gamma,
                from_logits=False,
            )
            losses.append(numpy_focal_loss_prob)

        for i, loss_1 in enumerate(losses):
            for loss_2 in losses[(i + 1):]:
                self.assertAllClose(loss_1, loss_2, atol=1e-5, rtol=1e-5)

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_PROB)
    def test_reduce_to_multiclass_crossentropy_from_probabilities(self, y_true,
                                                                  y_pred):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        focal_loss = sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                                   gamma=0)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true,
                                                             y_pred=y_pred)
        self.assertAllClose(focal_loss, ce)

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_LOGITS)
    def test_reduce_to_multiclass_crossentropy_from_logits(self, y_true,
                                                           y_pred):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        focal_loss = sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                                   gamma=0, from_logits=True)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.dtypes.cast(y_true, dtype=tf.dtypes.int64),
            logits=tf.dtypes.cast(y_pred, dtype=tf.dtypes.float32),
        )
        self.assertAllClose(focal_loss, ce)

    def _test_reduce_to_keras_loss(self, y_true, y_pred, from_logits: bool):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        keras_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits)
        focal_loss = SparseCategoricalFocalLoss(
            gamma=0, from_logits=from_logits)
        self.assertAllClose(keras_loss(y_true, y_pred),
                            focal_loss(y_true, y_pred))

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_LOGITS)
    def test_reduce_to_keras_loss_logits(self, y_true, y_pred):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        self._test_reduce_to_keras_loss(y_true, y_pred, from_logits=True)

    @named_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_PROB)
    def test_reduce_to_keras_loss_probabilities(self, y_true, y_pred):
        """Focal loss with gamma=0 should be the same as cross-entropy."""
        self._test_reduce_to_keras_loss(y_true, y_pred, from_logits=False)

    @named_parameters_with_testcase_names(
        n_examples=100, n_features=16, n_classes=[2, 3], epochs=2, gamma=[0, 2],
        from_logits=[True, False], random_state=np.random.RandomState(0))
    def test_train_dummy_multiclass_classifier(self, n_examples, n_features,
                                               n_classes, epochs, gamma,
                                               from_logits, random_state):
        # Generate some fake data
        x = random_state.binomial(n=n_classes, p=0.5,
                                  size=(n_examples, n_features))
        x = 2.0 * x / n_classes - 1.0
        weights = 100.0 * np.ones(shape=(n_features, n_classes))
        y = np.argmax(x.dot(weights), axis=-1)

        model = get_dummy_sparse_multiclass_classifier(
            n_features=n_features, n_classes=n_classes, gamma=gamma,
            from_logits=from_logits)
        history = model.fit(x, y, batch_size=n_examples, epochs=epochs,
                            callbacks=[tf.keras.callbacks.TerminateOnNaN()])

        # Check that we didn't stop early: if we did then we
        # encountered NaNs during training, and that shouldn't happen
        self.assertEqual(len(history.history['loss']), epochs)

        # Check that BinaryFocalLoss and binary_focal_loss agree (at
        # least when averaged)
        model_loss, *_ = model.evaluate(x, y)

        y_pred = model.predict(x)
        loss = sparse_categorical_focal_loss(y_true=y, y_pred=y_pred,
                                             gamma=gamma,
                                             from_logits=from_logits)
        loss = tf.math.reduce_mean(loss)
        self.assertAllClose(loss, model_loss)

    @named_parameters_with_testcase_names(gamma=[0, 1, 2],
                                          from_logits=[False, True])
    def test_get_config(self, gamma, from_logits):
        """Check the get_config() method."""
        loss1 = SparseCategoricalFocalLoss(gamma=gamma, from_logits=from_logits,
                                           name='focal_loss')
        config1 = loss1.get_config()
        loss2 = SparseCategoricalFocalLoss(**config1)
        config2 = loss2.get_config()
        self.assertEqual(config1, config2)

    @named_parameters_with_testcase_names(gamma=[0, 1, 2],
                                          from_logits=[False, True])
    def test_save_and_restore(self, gamma, from_logits):
        """Check if models compiled with focal loss can be saved/loaded."""
        model = get_dummy_sparse_multiclass_classifier(
            n_features=10, n_classes=3, gamma=gamma, from_logits=from_logits)
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

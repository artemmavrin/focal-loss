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
                                        from_logits=False, axis=-1):
    """Simple sparse categorical focal loss implementation using NumPy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if axis != -1:
        pred_dim = np.ndim(y_pred)
        axes = list(range(axis)) + list(range(axis + 1, pred_dim)) + [axis]
        y_pred = np.transpose(y_pred, axes)

    y_pred_shape_original = y_pred.shape
    n_classes = y_pred_shape_original[-1]
    y_true = np.reshape(y_true, newshape=[-1])
    y_pred = np.reshape(y_pred, newshape=[-1, n_classes])

    # One-hot encoding of integer labels
    y_true_one_hot = np.eye(n_classes)[y_true]

    if from_logits:
        y_pred = softmax(y_pred, axis=-1)
    else:
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)

    loss = -y_true_one_hot * (1 - y_pred) ** gamma * np.log(y_pred)
    loss = np.sum(loss, axis=-1)
    loss = np.reshape(loss, y_pred_shape_original[:-1])

    return loss


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
        from_logits=[True, False], random_state=np.random.default_rng(0))
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

    def test_with_higher_rank_inputs(self):
        """Addresses https://github.com/artemmavrin/focal-loss/issues/5"""

        def build_model():
            return tf.keras.Sequential([
                tf.keras.layers.Input((100, 10)),
                tf.keras.layers.GRU(13, return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(13)),
            ])

        x = np.zeros((20, 100, 10))
        y = np.ones((20, 100, 1))

        model = build_model()
        loss = SparseCategoricalFocalLoss(gamma=2)
        model.compile(loss=loss, optimizer='adam')
        model.fit(x, y)

    @named_parameters_with_testcase_names(axis=[0, 1, 2],
                                          from_logits=[False, True])
    def test_reduce_to_keras_with_higher_rank_and_axis(self, axis, from_logits):
        labels = tf.convert_to_tensor([[0, 1, 2], [0, 0, 0], [1, 1, 1]],
                                      dtype=tf.dtypes.int64)
        logits = tf.reshape(tf.range(27, dtype=tf.dtypes.float32),
                            shape=[3, 3, 3])
        probs = tf.nn.softmax(logits, axis=axis)

        y_pred = logits if from_logits else probs
        keras_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, y_pred, from_logits=from_logits, axis=axis)
        focal_loss = sparse_categorical_focal_loss(
            labels, y_pred, gamma=0, from_logits=from_logits, axis=axis)
        self.assertAllClose(focal_loss, keras_loss)

    @named_parameters_with_testcase_names(gamma=[0, 1, 2], axis=[0, 1, 2],
                                          from_logits=[False, True])
    def test_higher_rank_sanity_checks(self, gamma, axis, from_logits):
        labels = tf.convert_to_tensor([[0, 1, 2], [0, 0, 0], [1, 1, 1]],
                                      dtype=tf.dtypes.int64)
        logits = tf.reshape(tf.range(27, dtype=tf.dtypes.float32),
                            shape=[3, 3, 3])
        probs = tf.nn.softmax(logits, axis=axis)

        y_pred = logits if from_logits else probs
        numpy_loss = numpy_sparse_categorical_focal_loss(
            labels, y_pred, gamma=gamma, from_logits=from_logits, axis=axis)
        focal_loss = sparse_categorical_focal_loss(
            labels, y_pred, gamma=gamma, from_logits=from_logits, axis=axis)
        self.assertAllClose(focal_loss, numpy_loss)

    @named_parameters_with_testcase_names(gamma=[0, 1, 2],
                                          from_logits=[False, True])
    def test_with_dynamic_ranks(self, gamma, from_logits):
        # y_true must have defined rank
        y_true = tf.keras.backend.placeholder(None, dtype=tf.int64)
        y_pred = tf.keras.backend.placeholder((None, 2), dtype=tf.float32)
        with self.assertRaises(NotImplementedError):
            sparse_categorical_focal_loss(y_true, y_pred, gamma=gamma,
                                          from_logits=from_logits)

        # If axis is specified, y_pred must have a defined rank
        y_true = tf.keras.backend.placeholder((None,), dtype=tf.int64)
        y_pred = tf.keras.backend.placeholder(None, dtype=tf.float32)
        with self.assertRaises(ValueError):
            sparse_categorical_focal_loss(y_true, y_pred, gamma=gamma,
                                          from_logits=from_logits, axis=0)

        # It's fine if y_pred has undefined rank is axis=-1
        graph = tf.Graph()
        with graph.as_default():
            y_true = tf.keras.backend.placeholder((None,), dtype=tf.int64)
            y_pred = tf.keras.backend.placeholder(None, dtype=tf.float32)
            focal_loss = sparse_categorical_focal_loss(y_true, y_pred,
                                                       gamma=gamma,
                                                       from_logits=from_logits)

        labels = [0, 0, 1]
        logits = [[10., 0.], [5., -5.], [0., 10.]]
        probs = softmax(logits, axis=-1)

        pred = logits if from_logits else probs
        loss_numpy = numpy_sparse_categorical_focal_loss(
            labels, pred, gamma=gamma, from_logits=from_logits)

        with tf.compat.v1.Session(graph=graph) as sess:
            loss = sess.run(focal_loss,
                            feed_dict={y_true: labels, y_pred: pred})

        self.assertAllClose(loss, loss_numpy)



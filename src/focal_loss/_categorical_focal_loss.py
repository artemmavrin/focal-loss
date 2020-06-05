"""Multiclass focal loss implementation."""
#    __                          _     _
#   / _|                        | |   | |
#  | |_    ___     ___    __ _  | |   | |   ___    ___   ___
#  |  _|  / _ \   / __|  / _` | | |   | |  / _ \  / __| / __|
#  | |   | (_) | | (__  | (_| | | |   | | | (_) | \__ \ \__ \
#  |_|    \___/   \___|  \__,_| |_|   |_|  \___/  |___/ |___/

import tensorflow as tf


def sparse_categorical_focal_loss(y_true, y_pred, gamma, *,
                                  from_logits: bool = False) -> tf.Tensor:
    r"""Focal loss function for multiclass classification with integer labels.

    This loss function generalizes multiclass softmax cross-entropy by
    introducing a hyperparameter called the *focusing parameter* that allows
    hard-to-classify examples to be penalized more heavily relative to
    easy-to-classify examples.

    See :meth:`~focal_loss.binary_focal_loss` for a description of the focal
    loss in the binary setting, as presented in the original work [1]_.

    In the multiclass setting, with integer labels :math:`y`, focal loss is
    defined as

    .. math::

        L(y, \hat{\mathbf{p}})
        = -\left(1 - \hat{\mathbf{p}}_y\right)^\gamma \log(\hat{\mathbf{p}}_y)

    where

    *   :math:`y \in \{0, \ldots, K - 1\}` is an integer class label (:math:`K`
        denotes the number of classes),
    *   :math:`\hat{\mathbf{p}} \in [0, 1]^K` is a vector representing an
        estimated probability distribution over the :math:`K` classes,
    *   :math:`\gamma` (gamma, not :math:`y`) is the *focusing parameter* that
        specifies how much higher-confidence correct predictions contribute to
        the overall loss (the higher the :math:`\gamma`, the higher the rate at
        which easy-to-classify examples are down-weighted).

    The usual multiclass softmax cross-entropy loss is recovered by setting
    :math:`\gamma = 0`.

    Parameters
    ----------
    y_true : tensor-like, shape (N,)
        Integer class labels.

    y_pred : tensor-like, shape (N, K)
        Either probabilities or logits, depending on the `from_logits`
        parameter.

    gamma : float or tensor-like of shape (K,)
        The focusing parameter :math:`\gamma`. Higher values of `gamma` lead to
        easy-to-classify examples to contribute less to the loss relative to
        hard-to-classify examples. Must be non-negative. This can be a
        one-dimensional tensor, in which case it specifies a focusing parameter
        for each class.

    from_logits : bool, optional
        Whether `y_pred` contains logits or probabilities.

    Returns
    -------
    :class:`tf.Tensor`
        The focal loss for each example.

    Warnings
    --------
    This function does not reduce its output to a scalar, so it cannot be passed
    to :meth:`tf.keras.Model.compile` as a `loss` argument. Instead, use the
    wrapper class :class:`~focal_loss.SparseCategoricalFocalLoss`.

    References
    ----------
    .. [1] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár. Focal loss for
        dense object detection. IEEE Transactions on Pattern Analysis and
        Machine Intelligence, 2018.
        (`DOI <https://doi.org/10.1109/TPAMI.2018.2858826>`__)
        (`arXiv preprint <https://arxiv.org/abs/1708.02002>`__)

    See Also
    --------
    :meth:`~focal_loss.SparseCategoricalFocalLoss`
        A wrapper around this function that makes it a
        :class:`tf.keras.losses.Loss`.
    """
    gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
    scalar_gamma = gamma.shape == []

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int32)
    base_loss = tf.keras.backend.sparse_categorical_crossentropy(
        target=y_true, output=y_pred, from_logits=from_logits)

    if from_logits:
        probs = tf.nn.softmax(y_pred, axis=-1)
    else:
        probs = y_pred
    batch_size = tf.shape(y_true)[0]

    # For some reason y_true becomes shaped like (batch, 1) during training, so
    # the next line is a hack to ensure it's always rank 1 (needed for stacking)
    y_true = tf.cond(tf.rank(y_true) == 1, lambda: y_true, lambda: y_true[:, 0])

    indices = tf.stack([tf.range(batch_size), y_true], axis=1)
    probs = tf.gather_nd(probs, indices)

    if scalar_gamma:
        focal_modulation = (1 - probs) ** gamma
    else:
        focal_modulation = (1 - probs) ** tf.gather(gamma, y_true)

    return focal_modulation * base_loss


@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    r"""Focal loss function for multiclass classification with integer labels.

    This loss function generalizes multiclass softmax cross-entropy by
    introducing a hyperparameter called the *focusing parameter* that allows
    hard-to-classify examples to be penalized more heavily relative to
    easy-to-classify examples.

    This class is a wrapper around
    :class:`~focal_loss.sparse_categorical_focal_loss`. See the documentation
    there for details about this loss function.

    Parameters
    ----------
    gamma : float
        The focusing parameter :math:`\gamma`. Must be non-negative.

    from_logits : bool, optional
        Whether model prediction will be logits or probabilities.

    **kwargs : keyword arguments
        Other keyword arguments for :class:`tf.keras.losses.Loss` (e.g., `name`
        or `reduction`).

    See Also
    --------
    :meth:`~focal_loss.sparse_categorical_focal_loss`
        The function that performs the focal loss computation, taking a label
        tensor and a prediction tensor and outputting a loss.
    """
    def __init__(self, gamma, from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.from_logits = from_logits

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary containing the configuration of a
        layer. The same layer can be re-instantiated later (without its trained
        weights) from this configuration.

        Returns
        -------
        dict
            This layer's config.
        """
        config = super().get_config()
        config.update(gamma=self.gamma, from_logits=self.from_logits)
        return config

    def call(self, y_true, y_pred):
        """Compute the per-example focal loss.

        This method simply calls
        :meth:`~focal_loss.sparse_categorical_focal_loss` with the appropriate
        arguments.

        Parameters
        ----------
        y_true : tensor-like
            Binary (0 or 1) class labels.

        y_pred : tensor-like
            Either probabilities or logits, depending on the `from_logits`
            attribute.

        Returns
        -------
        :class:`tf.Tensor`
            The per-example focal loss. Reduction to a scalar is handled by
            this layer's
            :meth:`~focal_loss.SparseCateogiricalFocalLoss.__call__` method.
        """
        return sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                             gamma=self.gamma,
                                             from_logits=self.from_logits)

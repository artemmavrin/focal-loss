==========
Focal Loss
==========

.. image:: https://img.shields.io/pypi/pyversions/focal-loss
    :target: https://pypi.org/project/focal-loss
    :alt: Python Version

.. image:: https://img.shields.io/pypi/v/focal-loss
    :target: https://pypi.org/project/focal-loss
    :alt: PyPI Package Version

.. image:: https://img.shields.io/github/last-commit/artemmavrin/focal-loss/master
    :target: https://github.com/artemmavrin/focal-loss
    :alt: Last Commit

.. image:: https://github.com/artemmavrin/focal-loss/workflows/Python%20package/badge.svg
    :target: https://github.com/artemmavrin/focal-loss/actions?query=workflow%3A%22Python+package%22
    :alt: GitHub Actions Build Status

.. image:: https://codecov.io/gh/artemmavrin/focal-loss/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/artemmavrin/focal-loss
    :alt: Code Coverage

.. image:: https://readthedocs.org/projects/focal-loss/badge/?version=latest
    :target: https://focal-loss.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/artemmavrin/focal-loss
    :target: https://github.com/artemmavrin/focal-loss/blob/master/LICENSE
    :alt: License

TensorFlow implementation of focal loss: a loss function generalizing binary and
multiclass cross-entropy loss that penalizes hard-to-classify examples.

.. image:: images/focal-loss.png
    :scale: 40 %
    :alt: Focal loss plot
    :align: center

The :mod:`focal_loss` package provides functions and classes that can be used as
off-the-shelf replacements for :mod:`tf.keras.losses` functions and classes,
respectively.

.. code-block:: python

    # Typical tf.keras API usage
    import tensorflow as tf
    from focal_loss import BinaryFocalLoss

    model = tf.keras.Model(...)
    model.compile(
        optimizer=...,
        loss=BinaryFocalLoss(gamma=2),  # Used here like a tf.keras loss
        metrics=...,
    )
    history = model.fit(...)

The :mod:`focal_loss` package includes the functions

* :meth:`~focal_loss.binary_focal_loss`
* :meth:`~focal_loss.sparse_categorical_focal_loss`

and wrapper classes

* :class:`~focal_loss.BinaryFocalLoss` (use like :class:`tf.keras.losses.BinaryCrossentropy`)
* :class:`~focal_loss.SparseCategoricalFocalLoss` (use like :class:`tf.keras.losses.SparseCategoricalCrossentropy`)

.. toctree::
    :caption: Contents
    :maxdepth: 1

    install
    api
    Source Code on GitHub <https://github.com/artemmavrin/focal-loss>

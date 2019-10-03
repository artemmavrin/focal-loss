==========
Focal Loss
==========

.. image:: https://img.shields.io/pypi/pyversions/focal-loss
    :target: https://pypi.org/project/focal-loss
    :alt: Python Version

.. image:: https://img.shields.io/pypi/v/focal-loss
    :target: https://pypi.org/project/focal-loss
    :alt: PyPI Package Version

.. image:: https://travis-ci.com/artemmavrin/focal-loss.svg?branch=master
    :target: https://travis-ci.com/artemmavrin/focal-loss
    :alt: Build Status

.. image:: https://codecov.io/gh/artemmavrin/focal-loss/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/artemmavrin/focal-loss
    :alt: Code Coverage

.. image:: https://readthedocs.org/projects/focal-loss/badge/?version=latest
    :target: https://focal-loss.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/artemmavrin/focal-loss
    :target: https://github.com/artemmavrin/focal-loss/blob/master/LICENSE
    :alt: License

TensorFlow implementation of focal loss: a loss function generalizing binary
cross-entropy loss that penalizes hard-to-classify examples.

.. image:: images/focal-loss.png
    :scale: 40 %
    :alt: Focal loss plot
    :align: center

The :mod:`focal_loss` package provides a function
:meth:`~focal_loss.binary_focal_loss` and a class
:class:`~focal_loss.BinaryFocalLoss` that can be used as stand-in replacements
for :mod:`tf.keras.losses` functions and classes, respectively.

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

.. toctree::
    :caption: Contents
    :maxdepth: 1

    install
    api
    Source Code on GitHub <https://github.com/artemmavrin/focal-loss>

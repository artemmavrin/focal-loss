"""Unit tests for the parameter validation helper functions."""

import numpy as np
import pytest

from focal_loss.utils.validation import (check_bool, check_int, check_float,
                                         check_type)


def test_check_bool():
    """Sanity checks for check_bool()."""
    assert check_bool(True)
    assert not check_bool(False)
    assert check_bool(None, allow_none=True) is None
    for bad in (None, 0, 1, 'a', [1, 2, 3], []):
        with pytest.raises(TypeError):
            check_bool(bad)
    assert check_bool(None, allow_none=True, default=True)


def test_check_int():
    """Sanity checks for check_int()."""
    assert check_int(1) == 1
    assert check_int(np.int_(2)) == np.int_(2)
    assert check_int(None, allow_none=True) is None
    for bad in (None, 0., 1., 'a', [1, 2, 3], []):
        with pytest.raises(TypeError):
            check_int(bad)

    # Check that positivity constraints are enforced
    assert check_int(1, positive=True) == 1
    with pytest.raises(ValueError):
        check_int(0, positive=True)

    # Check that minimum value constraints are enforced
    assert check_int(0, minimum=0) == 0
    assert check_int(1, minimum=0) == 1
    with pytest.raises(ValueError):
        check_int(-1, minimum=0)

    # Check that maximum value constraints are enforced
    assert check_int(0, maximum=1) == 0
    assert check_int(1, maximum=1) == 1
    with pytest.raises(ValueError):
        check_int(2, maximum=1)

    # Check that default values are honored
    assert check_int(None, allow_none=True, default=1) == 1

    def func(a, b):
        a = check_int(a, name='a', positive=True, minimum=2, maximum=10)
        b = check_int(b, positive=True, minimum=2, maximum=10)
        return a + b

    assert func(5, 5) == 10
    for a_bad in (0, 1, 11):
        with pytest.raises(ValueError):
            func(a_bad, 5)
    for b_bad in (0, 1, 11):
        with pytest.raises(ValueError):
            func(5, b_bad)


def test_check_float():
    """Sanity checks for check_float()."""
    assert check_float(1.0) == 1.0
    assert check_float(np.float32(2)) == np.float32(2)
    assert check_float(np.float64(2)) == np.float64(2)
    assert check_float(None, allow_none=True) is None
    for bad in (None, 'a', [1, 2, 3], []):
        with pytest.raises(TypeError):
            check_float(bad)

    # Check that positivity constraints are enforced
    assert check_float(1, positive=True) == 1.0
    with pytest.raises(ValueError):
        check_float(0, positive=True)

    # Check that minimum value constraints are enforced
    assert check_float(0, minimum=0) == 0.0
    assert check_float(1, minimum=0) == 1.0
    with pytest.raises(ValueError):
        check_float(-1, minimum=0)

    # Check that maximum value constraints are enforced
    assert check_float(0, maximum=1) == 0.0
    assert check_float(1, maximum=1) == 1.0
    with pytest.raises(ValueError):
        check_float(2, maximum=1)

    # Check that default values are honored
    assert check_float(None, allow_none=True, default=1.0) == 1.0

    def func(a, b):
        a = check_float(a, name='a', positive=True, minimum=2, maximum=10)
        b = check_float(b, positive=True, minimum=2, maximum=10)
        return a + b

    assert func(5, 5) == 10.0
    for a_bad in (0, 1, 11):
        with pytest.raises(ValueError):
            func(a_bad, 5)
    for b_bad in (0, 1, 11):
        with pytest.raises(ValueError):
            func(5, b_bad)


def test_check_type():
    """Sanity checks for check_type()."""
    assert check_type(1, int) == 1
    with pytest.raises(TypeError):
        check_type(1, str)
    with pytest.raises(TypeError):
        check_type(1, str, name='num')
    assert check_type(1, int, func=str) == '1'
    with pytest.raises(ValueError):
        check_type(1, int, func='not callable')
    with pytest.raises(TypeError):
        check_type(2.0, str, error_message='Not a string!')
    assert check_type(None, str, allow_none=True) is None
    assert check_type(None, int, allow_none=True, default=0) == 0

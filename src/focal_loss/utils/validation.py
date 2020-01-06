"""Helper functions for function parameter validation."""

import numbers


def check_type(obj, base, *, name=None, func=None, allow_none=False,
               default=None, error_message=None):
    """Check whether an object is an instance of a base type.

    Parameters
    ----------
    obj : object
        The object to be validated.

    name : str
        The name of `obj` in the calling function.

    base : type or tuple of type
        The base type that `obj` should be an instance of.

    func: callable, optional
        A function to be applied to `obj` if it is of type `base`. If None, no
        function will be applied and `obj` will be returned as-is.

    allow_none : bool, optional
        Indicates whether the value None should be allowed to pass through.

    default : object, optional
        The default value to return if `obj` is None and `allow_none` is True.
        If `default` is not None, it must be of type `base`, and it will have
        `func` applied to it if `func` is not None.

    error_message : str or None, optional
        Custom error message to display if the type is incorrect.

    Returns
    -------
    base type or None
        The validated object.

    Raises
    ------
    TypeError
        If `obj` is not an instance of `base`.

    Examples
    --------
    >>> check_type(1, int)
    1
    >>> check_type(1, (int, str))
    1
    >>> check_type(1, str)
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: str. Actual: int.
    >>> check_type(1, (str, bool))
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: (str, bool). Actual: int.
    >>> print(check_type(None, str, allow_none=True))
    None
    >>> check_type(1, str, name='num')
    Traceback (most recent call last):
    ...
    TypeError: Invalid type for parameter 'num'. Expected: str. Actual: int.
    >>> check_type(1, int, func=str)
    '1'
    >>> check_type(1, int, func='not callable')
    Traceback (most recent call last):
    ...
    ValueError: Parameter 'func' must be callable or None.
    >>> check_type(2.0, str, error_message='Not a string!')
    Traceback (most recent call last):
    ...
    TypeError: Not a string!
    >>> check_type(None, int, allow_none=True, default=0)
    0

    """
    if allow_none and obj is None:
        if default is not None:
            return check_type(default, base=base, name=name, func=func,
                              allow_none=False)
        return None

    if isinstance(obj, base):
        if func is None:
            return obj
        elif callable(func):
            return func(obj)
        else:
            raise ValueError('Parameter \'func\' must be callable or None.')

    # Handle wrong type
    if isinstance(base, tuple):
        expect = '(' + ', '.join(cls.__name__ for cls in base) + ')'
    else:
        expect = base.__name__
    actual = type(obj).__name__
    if error_message is None:
        error_message = 'Invalid type'
        if name is not None:
            error_message += f' for parameter \'{name}\''
        error_message += f'. Expected: {expect}. Actual: {actual}.'
    raise TypeError(error_message)


def check_bool(obj, *, name=None, allow_none=False, default=None):
    """Validate boolean function arguments.

    Parameters
    ----------
    obj : object
        The object to be validated.

    name : str, optional
        The name of `obj` in the calling function.

    allow_none : bool, optional
        Indicates whether the value None should be allowed.

    default : object, optional
        The default value to return if `obj` is None and `allow_none` is True.

    Returns
    -------
    bool or None
        The validated bool.

    Raises
    ------
    TypeError
        If `obj` is not an instance of bool.

    Examples
    --------
    >>> check_bool(True)
    True
    >>> check_bool(1.0)
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: bool. Actual: float.
    >>> a = (1 < 2)
    >>> check_bool(a, name='a')
    True
    >>> b = 'not a bool'
    >>> check_bool(b, name='b')
    Traceback (most recent call last):
    ...
    TypeError: Invalid type for parameter 'b'. Expected: bool. Actual: str.
    """
    return check_type(obj, name=name, base=bool, func=bool,
                      allow_none=allow_none, default=default)


def _check_numeric(*, check_func, obj, name, base, func, positive, minimum,
                   maximum, allow_none, default):
    """Helper function for check_float and check_int."""
    obj = check_type(obj, name=name, base=base, func=func,
                     allow_none=allow_none, default=default)

    if obj is None:
        return None

    positive = check_bool(positive, name='positive')
    if positive and obj <= 0:
        if name is None:
            message = 'Parameter must be positive.'
        else:
            message = f'Parameter \'{name}\' must be positive.'
        raise ValueError(message)

    if minimum is not None:
        minimum = check_func(minimum, name='minimum')
        if obj < minimum:
            if name is None:
                message = f'Parameter must be at least {minimum}.'
            else:
                message = f'Parameter \'{name}\' must be at least {minimum}.'
            raise ValueError(message)

    if maximum is not None:
        maximum = check_func(maximum, name='minimum')
        if obj > maximum:
            if name is None:
                message = f'Parameter must be at most {maximum}.'
            else:
                message = f'Parameter \'{name}\' must be at most {maximum}.'
            raise ValueError(message)

    return obj


def check_int(obj, *, name=None, positive=False, minimum=None, maximum=None,
              allow_none=False, default=None):
    """Validate integer function arguments.

    Parameters
    ----------
    obj : object
        The object to be validated.

    name : str, optional
        The name of `obj` in the calling function.

    positive : bool, optional
        Whether `obj` must be a positive integer (1 or greater).

    minimum : int, optional
        The minimum value that `obj` can take (inclusive).

    maximum : int, optional
        The maximum value that `obj` can take (inclusive).

    allow_none : bool, optional
        Indicates whether the value None should be allowed.

    default : object, optional
        The default value to return if `obj` is None and `allow_none` is True.

    Returns
    -------
    int or None
        The validated integer.

    Raises
    ------
    TypeError
        If `obj` is not an integer.

    ValueError
        If any of the optional positivity or minimum and maximum value
        constraints are violated.

    Examples
    --------
    >>> check_int(0)
    0
    >>> check_int(1, positive=True)
    1
    >>> check_int(1.0)
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: Integral. Actual: float.
    >>> check_int(-1, positive=True)
    Traceback (most recent call last):
    ...
    ValueError: Parameter must be positive.
    >>> check_int(1, name='a', minimum=10)
    Traceback (most recent call last):
    ...
    ValueError: Parameter 'a' must be at least 10.

    """
    return _check_numeric(check_func=check_int, obj=obj, name=name,
                          base=numbers.Integral, func=int, positive=positive,
                          minimum=minimum, maximum=maximum,
                          allow_none=allow_none, default=default)


def check_float(obj, *, name=None, positive=False, minimum=None, maximum=None,
                allow_none=False, default=None):
    """Validate float function arguments.

    Parameters
    ----------
    obj : object
        The object to be validated.

    name : str, optional
        The name of `obj` in the calling function.

    positive : bool, optional
        Whether `obj` must be a positive float.

    minimum : float, optional
        The minimum value that `obj` can take (inclusive).

    maximum : float, optional
        The maximum value that `obj` can take (inclusive).

    allow_none : bool, optional
        Indicates whether the value None should be allowed.

    default : object, optional
        The default value to return if `obj` is None and `allow_none` is True.

    Returns
    -------
    float or None
        The validated float.

    Raises
    ------
    TypeError
        If `obj` is not a float.

    ValueError
        If any of the optional positivity or minimum and maximum value
        constraints are violated.

    Examples
    --------
    >>> check_float(0)
    0.0
    >>> check_float(1.0, positive=True)
    1.0
    >>> check_float(1.0 + 1.0j)
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: Real. Actual: complex.
    >>> check_float(-1, positive=True)
    Traceback (most recent call last):
    ...
    ValueError: Parameter must be positive.
    >>> check_float(1.2, name='a', minimum=10)
    Traceback (most recent call last):
    ...
    ValueError: Parameter 'a' must be at least 10.0.

    """
    return _check_numeric(check_func=check_float, obj=obj, name=name,
                          base=numbers.Real, func=float, positive=positive,
                          minimum=minimum, maximum=maximum,
                          allow_none=allow_none, default=default)

from functools import wraps


def cache(func):
    """
    Decorator for caching return values of functions.
    This is very similar to `functools.cache`, but without the memory leaks (see
    https://docs.astral.sh/ruff/rules/cached-instance-method/).

    Example:

    .. code-block:: python

        num_calls = 0

        @cache
        def increment(x):
            num_calls += 1
            return x + 1

        increment(0)  # returns 1, num_calls = 1
        increment(1)  # returns 2, num_calls = 2
        increment(0)  # returns 1, num_calls = 2


    Args:
        func (function): The function you want to cache the return value of

    Returns:
        return value of func
    """
    attr_cache = f"_{func.__name__}_cache"

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, attr_cache):
            setattr(self, attr_cache, {})

        cache = getattr(self, attr_cache)

        key = (args, frozenset(kwargs.items()))
        if key in cache:
            return cache[key]
        result = func(self, *args, **kwargs)
        cache[key] = result
        return result

    return wrapper

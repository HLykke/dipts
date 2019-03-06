import inspect

__all__ = ['validate_io_types']

def validate_io_types(f):
    """Decorator function that validates input and return types
    if specified like example below:

    @validate_io_types
    def f(x: int = 2, y: float = np.pi, z: tuple = (2,4,5) -> float:
        ...

    """

    def validate(*args, **kwargs):
        specs = inspect.getfullargspec(f).annotations

        #  Set aside the return value to after calculation is done
        return_type = specs['return']
        del specs['return']

        #  Assert argument(s)
        for i, (v, vtype) in enumerate(specs.items()):
            # print("\t-  %s, of type %s: %s\n"%(str(name), str(vtype), str(v)))
            v = args[i]
            try:
                assert isinstance(v, vtype)
            except AssertionError:
                raise ValueError(
                    "Variable: '%s', function: '%s' is not of type: %s\n" % (str(v), f.__name__, str(vtype)))

        # Assert return value(s)
        res = f(*args, **kwargs)
        if res is not None:
            try:
                assert isinstance(res, return_type)
            except AssertionError:
                raise ValueError("Return value from function: '%s' is not of type: %s\n" % (f.__name__, return_type))

        print("-------------------- IO-types of '%s' OK --------------------" % (f.__name__))
        return res

    return validate


def test_validate_io_types():
    @validate_io_types
    def f1(x: int, y: tuple) -> int:
        return x

    success = {}
    try:
        f1(1, (2, 3, 4))
        success.update({'Everything fine': True})
    except Exception:
        success.update({'Everything fine': False})

    try:
        f1(3.3, (1, 2, 3))
        success.update({'Reaction when float is given, while int was expected': False})
    except Exception:
        success.update({'Reaction when float is given, while int was expected': True})

    try:
        f1(3, 3)
        success.update({'Reaction when int is given, while tuple was expected': False})
    except Exception:
        success.update({'Reaction when int is given, while tuple was expected': True})

    for key, value in success.items():
        assert value, key


def test():
    test_validate_io_types()


if __name__ == '__main__':
    test()

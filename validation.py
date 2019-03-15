import inspect
from typing import Iterable, Tuple
import warnings

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
        print("Full specs of io to function: ", specs)

        try:
            print("!!!!!!!!!!!!!!!!!is it self? :", args[0].__self__)
            del args[0]
        except Exception:
            print("Arg that is not self: ", args[0])
        # del args[0]


        #  Set aside the return value to after calculation is done
        try:
            return_type = specs['return']
            del specs['return']
            typed_return = True
        except KeyError:
            typed_return = False
        #  Assert argument(s)
        for i, (v, vtype) in enumerate(specs.items()):
            # print("\t-  %s, of type %s: %s\n"%(str(name), str(vtype), str(v)))
            print(args, i)
            print(v)
            v = args[i]
            if str(vtype).find('[')!=-1:
                warnings.warn('\nTyping warning: Parametrized parameter-types are not yet supported for type-checking.')
            else:
                try:
                    print("Instance: ", v, "type: ", vtype, 'res :', isinstance(v, vtype))
                    assert isinstance(v, vtype)
                except AssertionError:
                    raise ValueError("Variable: '%s', function: '%s' is not of type: %s\n" % (str(v), f.__name__, str(vtype)))

        # Assert return value(s)
        res = f(*args, **kwargs)
        if typed_return:
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
    def f1(x: int, y: Tuple) -> int:
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
        print('\t', key, ': ', value)
        assert value, key

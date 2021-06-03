import yaml
import types


class RecursiveNamespace(types.SimpleNamespace):
    """RecursiveNamespace

    Instantiating a RecursiveNamedspace (SimpleNamespace) on a dictionary creates a new attribute on the namedspace
    class for each key in the dictionary.

    For each value that is a dict, another RecursiveNamespace is instantiated on it.

    The elements of lists and tuples are also checked.
    """

    def __init__(self, /, **kwargs):
        """Create a SimpleNamespace recursively on the objects of the **kwargs."""
        self.__dict__.update({k: self.__elt(v) for k, v in kwargs.items()})

    def __elt(self, elt):
        """Recurse into elt to create leaf namepace objects"""
        if type(elt) is dict:
            return RecursiveNamespace(**elt)
        if type(elt) in (list, tuple):
            return [self.__elt(i) for i in elt]
        return elt

    def to_dict(self):
        raise NotImplementedError


def read_yaml_file(filepath, return_namespace=True):
    """Read a .yaml file to a RecursiveNamespace"""
    with open(filepath, "r") as stream:
        try:
            output = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
    return RecursiveNamespace(**output) if return_namespace else output


def read_json_file(filepath, return_namespace=True):
    """Read a .json file to a RecursiveNamedspace"""
    raise NotImplementedError()

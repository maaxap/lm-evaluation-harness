import builtins

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("lowercase")
class LowercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("uppercase")
class UppercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("map")
class MapFilter(Filter):
    def __init__(self, mapping_dict: dict = None, default_value=None) -> None:
        """
        Initializes the MapFilter with a given mapping dictionary and default value.

        Args:
        - mapping_dict (dict): A dictionary containing the key-value mappings.
                               Default is an empty dictionary.
        - default_value (Any): The value to be returned when a key is not found in the mapping_dict.
                               Default is None.

        Example:
        mapper = MapFilter({'A': 1, 'B': 2}, default_value=0)
        """
        if mapping_dict is None:
            mapping_dict = {}
        assert isinstance(
            mapping_dict, dict
        ), "Provided mapping_dict is not a dictionary"
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("cast_dtype")
class CastDtypeFilter(Filter):
    def __init__(self, dtype: str) -> None:
        """
        Initializes the CastDtypeFilter with a given data type.

        Args:
        - dtype (str): The data type to cast the responses to.
                       Must be a valid built-in Python data type.

        Example:
        caster = CastDtypeFilter('int')
        """
        assert dtype in {
            "int",
            "float",
            "str",
            "bool",
            "complex",
            "bytes",
            "bytearray",
            "memoryview",
            "list",
            "tuple",
            "set",
            "frozenset",
            "dict",
        }, f"Provided dtype is not a valid built-in Python data type: {dtype}"
        self.dtype = getattr(builtins, dtype, None)

    def apply(self, resps, docs):
        def cast_dtype(inst):
            return [self.dtype(resp) for resp in inst]

        return [cast_dtype(resp) for resp in resps]

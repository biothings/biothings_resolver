from typing import Iterable, Callable, Any, Optional


def dict_get_nested(d: dict, k: str) -> Any:
    for key in k.split('.'):
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d


def transform_str(s: str, transforms: Iterable[Callable[[str], Optional[str]]]) -> Optional[str]:
    for transform in transforms:
        if s is None:
            return None  # stop when s is no longer a string
        s = transform(s)
    return s


def multiple_fields(fields: Iterable[str]) -> Callable[[dict], str]:
    def f(doc: dict) -> str:
        for field in fields:
            id_value = dict_get_nested(doc, field)
            if isinstance(id_value, str):
                return id_value
    return f

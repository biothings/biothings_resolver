import re
from re import Pattern
from typing import Any, Callable, Optional, Union


def filter_nan(id_value: Any) -> Optional[str]:
    if id_value != id_value:
        return None
    else:
        return id_value


def filter_pubchem_cid(id_value: str) -> Optional[str]:
    if isinstance(id_value, str) and id_value.startswith('CID') and len(id_value) == 12:
        return str(int(id_value[4:]))
    else:
        return id_value


def filter_on_regex(regex: Union[str, Pattern]) -> Callable[[str], Optional[str]]:
    regex_compiled = re.compile(regex)

    def f(id_value: str):
        if regex_compiled.match(id_value):
            return None
        else:
            return id_value
    return f

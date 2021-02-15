import collections.abc
import warnings

from typing import Dict, Iterator, Optional, List, Tuple, Set, Iterable, \
    Generator
from collections import OrderedDict

from .curie import validate_prefix


class CanonDict(collections.abc.MutableMapping):
    def __init__(self, *args, **kwargs):
        self._dict = {}
        self._ci = False
        self._alias_mapping = {}
        self._cf_mapping = {}
        input_dict = dict(*args, **kwargs)
        for k, v in input_dict.items():
            self[k] = v

    @property
    def case_sensitive(self) -> bool:
        return not self._ci

    @case_sensitive.setter
    def case_sensitive(self, v: bool):
        old_ci = self._ci
        self._ci = not v
        new_cf_mapping = {}
        for k in set(self._dict) | set(self._alias_mapping):
            cf_k = self._fold_case(k)
            if cf_k in new_cf_mapping:
                self._ci = old_ci
                raise RuntimeError(f"{k}, {new_cf_mapping[cf_k]} both exist")
            new_cf_mapping[cf_k] = k
        self._cf_mapping = new_cf_mapping

    def add_alias(self, alias, key):
        cf_alias = self._fold_case(alias)
        if cf_alias in self._cf_mapping:
            raise KeyError(f"alias {self._cf_mapping[cf_alias]} already exist")
        if key not in self._dict:
            raise KeyError(f"{key}")
        self._alias_mapping[alias] = key
        self._cf_mapping[self._fold_case(alias)] = alias

    def delete_alias(self, alias):
        cf_alias = self._fold_case(alias)
        del self._alias_mapping[alias]
        del self._cf_mapping[cf_alias]

    def _fold_case(self, k):
        # FIXME: inconsistent behavior/may depend on locale
        if self._ci:
            try:
                return k.lower()
            except (AttributeError, TypeError):
                pass
        return k

    def get_canon_key(self, k):
        try:
            # if we already have this key, overwrite with the casing we have
            k = self._cf_mapping[self._fold_case(k)]
            # and if it is an alias, get the canonical version
            k = self._alias_mapping[k]
        except KeyError:
            pass  # if either step above has failed, it doesn't matter
        return k

    def __setitem__(self, k, v):
        k = self.get_canon_key(k)
        if k not in self._dict:
            self._cf_mapping[self._fold_case(k)] = k
        self._dict[k] = v

    def __delitem__(self, k):
        cased_k = self._cf_mapping[self._fold_case(k)]
        canon_k = self.get_canon_key(k)
        del self._dict[canon_k]
        # trying to delete via an alias, then delete the alias as well
        if cased_k in self._alias_mapping:
            del self._alias_mapping[cased_k]

    def __getitem__(self, k):
        k = self.get_canon_key(k)
        return self._dict[k]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator:
        return iter(self._dict)

    def __repr__(self):
        return f"CanonDict: {repr(self._dict)}"

    def __str__(self):
        return f"CanonDict: {str(self._dict)}"


class CPDict(CanonDict):
    """Case Preserving Dictionary"""
    def __init__(self, *args, **kwargs):
        super(CPDict, self).__init__(*args, **kwargs)
        self.case_sensitive = False


class LRU(collections.abc.MutableMapping):
    def __init__(self, maxsize=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxsize = maxsize
        self._od = OrderedDict()

    def __delitem__(self, k) -> None:
        del self._od[k]

    def __len__(self) -> int:
        return len(self._od)

    def __iter__(self):
        return iter(self._od)

    def __getitem__(self, key):
        value = self._od[key]
        self._od.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self) > self.maxsize:
            self._od.popitem(last=False)

    def clear(self):
        self._od.clear()

import copy
import re

from typing import (Dict, Iterable, Optional, Union, List, Pattern, Callable,
                    Collection, Mapping, Any)

import biothings_client

from .utils import dict_get_nested


class ResolverAgent:
    """
    Base Agent Class
    """
    def __init__(self):
        super().__init__()
    
    @property
    def input_type(self) -> str:
        raise NotImplementedError
    
    @property
    def output_types(self) -> List[str]:
        raise NotImplementedError

    @property
    def max_batch_size(self) -> Optional[int]:
        raise NotImplementedError
    
    def lookup(self, input_values: Collection, output_types: Collection[str]) \
            -> List[Mapping[str, list]]:
        """
        Lookup Output Values

        To Agent Implementers: Because input value may be non-hashable,
        it has been decided that the output will be in the same order as input
        so that non-hashable inputs can be handled.
        """
        raise NotImplementedError


class RegExAgent(ResolverAgent):
    def __init__(self, from_regex: Union[str, Pattern],
                 to_regex: Union[str, Pattern, Callable],
                 input_type: str, output_types: List[str]):
        super().__init__()
        self.from_regex = from_regex
        self.to_regex = to_regex
        self._in_type = input_type
        self._out_types = output_types.copy()

    @property
    def max_batch_size(self) -> Optional[int]:
        return None

    @property
    def input_type(self) -> str:
        return self._in_type

    @property
    def output_types(self) -> List[str]:
        return self._out_types.copy()

    def lookup(self, input_values: Collection[str],
               output_types) -> List[Dict[str, List[str]]]:
        out_list = []
        output_types = list(output_types)
        for in_value in input_values:
            # re.sub auto caches compiled patterns
            out_value = re.sub(self.from_regex, self.to_regex, in_value)
            out = {}
            for k in output_types:
                out[k] = [out_value]
                # this way it is a different list with same content
            out_list.append(out)
        return out_list


class BioThingsAPIAgent(ResolverAgent):
    def __init__(self, biothings_type: str,
                 input_type: str,
                 search_scope: Collection[str],
                 output_fields: Mapping[str, Collection[str]]):

        super().__init__()
        self.client: biothings_client.BiothingClient = \
            biothings_client.get_client(biothings_type)
        self._in_type = input_type
        self._out_types = list(output_fields.keys())
        self.search_scope = list(search_scope)
        # forward mapping to build the list of fields during search
        self.out_fwd_map = {}
        for k, v in output_fields.items():
            self.out_fwd_map[k] = set(v)  # use set for de-dup
        self.out_back_map = {}
        for k, v in self.out_fwd_map.items():
            for field_name in v:
                if field_name in self.out_back_map:
                    raise ValueError("one field used in multiple output types")
                self.out_back_map[field_name] = k

    @property
    def input_type(self) -> str:
        return self._in_type

    @property
    def output_types(self) -> List[str]:
        return self._out_types.copy()

    @property
    def max_batch_size(self) -> Optional[int]:
        return 1000

    def lookup(self, input_values: Collection, output_types: Collection[str]) \
            -> List[Dict[str, list]]:
        fields_set = set()
        for output_type in output_types:
            fields_set.union(self.out_fwd_map[output_type])
        search_fields = list(fields_set)
        search_items = list(input_values)
        items_idx = {}
        for idx, item in enumerate(search_items):
            items_idx.setdefault(str(item), []).append(idx)
            # we know this would be str

        results = self.client.querymany(search_items, scopes=self.search_scope,
                                        fields=search_fields, verbose=False)
        final_output = [{} for _ in range(len(search_items))]
        # so that users who attempt to modify w/o copy will not be screwed
        for result in results:
            k = result.get('query')
            if k is None:
                continue  # bad record
            elif str(k) in items_idx:
                if result.get('notfound', False):
                    continue
                    # resume the loop, we don't have what we want, leave as {}
                else:
                    out = {}
                    for value_key, out_type in self.out_back_map.items():
                        lookup_value = dict_get_nested(result, value_key)
                        if lookup_value is not None:
                            out.setdefault(out_type, []).append(lookup_value)
                        else:  # did not find value, continue to next
                            continue
                    for idx in items_idx[str(k)]:
                        final_output[idx] = copy.deepcopy(out)
                        # same document, multiple copies
            else:
                pass  # was here for debugging when query was not str
        return final_output

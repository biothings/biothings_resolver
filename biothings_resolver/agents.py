import re
from typing import Dict, Iterable, Optional, Union, List, Pattern, Callable

import biothings_client

from .utils import dict_get_nested


class IDLookupAgent:
    def __init__(self):
        super(IDLookupAgent, self).__init__()

    def lookup(self, id_values: Iterable[str]) -> Dict[str, Optional[str]]:
        raise NotImplementedError


class BioThingsAPIAgent(IDLookupAgent):
    def __init__(self, biothings_type: str,
                 search_scope: Union[List[str], str],
                 value_fields: Union[List[str], str]):

        super().__init__()
        self.client: biothings_client.BiothingClient = biothings_client.get_client(biothings_type)
        self.search_scope = search_scope if type(search_scope) is list else [search_scope]
        self.value_fields = value_fields if type(value_fields) is list else [value_fields]

    def lookup(self, id_values: Iterable[str]) -> Dict[str, Optional[str]]:
        final_result = {}
        id_set = set(str(i) for i in id_values)  # speeds up __contains__
        # here, we can replace with requests
        results = self.client.querymany(list(id_set), scopes=self.search_scope,
                                        fields=self.value_fields, verbose=False)
        for result in results:
            k = result.get('query')
            if k is None:
                continue  # bad record
            elif k in id_set:
                if result.get('notfound', False):
                    final_result[k] = None
                    continue  # resume the loop, we don't have what we want
                else:
                    output = set()
                    for value_key in self.value_fields:
                        lookup_value = dict_get_nested(result, value_key)
                        if lookup_value is not None:
                            output.add(str(lookup_value))
                        else:  # did not find value, continue to next
                            continue
                    if len(output) == 0:
                        final_result[k] = None
                    elif len(output) == 1:
                        final_result[k] = output.pop()
                    else:
                        final_result[k] = list(output)
            else:
                pass  # was here for debugging when query was not str
        for missing_key in (id_set - set(final_result.keys())):
            # in the weird case that the API did not reply our query
            final_result[missing_key] = None
        return final_result


class RegExAgent(IDLookupAgent):
    def __init__(self, from_regex: Union[str, Pattern],
                 to_regex: Union[str, Pattern, Callable]):
        super().__init__()
        self.from_regex = from_regex
        self.to_regex = to_regex

    def lookup(self, id_values: Iterable[str]) -> Dict[str, str]:
        out = {}
        for id_value in id_values:
            out[id_value] = re.sub(self.from_regex, self.to_regex, id_value)
            # re.sub auto caches compiled patterns
        return out

import copy
import collections.abc
import warnings

from typing import Any, Dict, Iterator, Optional, List, Tuple, Set, Iterable, \
    Generator
from collections import OrderedDict

from .agents import IDLookupAgent
from .biolink_model import prefixes as biolink_prefixes
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


class BiolinkModelPrefixes(collections.abc.Mapping):
    # TODO: merge this and CPDict, these two are pretty similar
    def __init__(self, case_insensitive: bool = True):
        self._canonical_prefixes = copy.deepcopy(biolink_prefixes)
        self._mapping = {}
        self._case_insensitive = case_insensitive
        self._setup()

    def _setup(self):
        canon_maps = [
            ('FB', 'FlyBase'),
            ('WB', 'WormBase'),
            ('DBSNP', 'dbSNP'),
        ]
        for alt_form, canon in canon_maps:
            try:
                self._canonical_prefixes.pop(alt_form)
                if self._case_insensitive:
                    self._mapping[alt_form.lower()] = canon
                else:
                    self._mapping[alt_form] = canon
            except KeyError:
                pass
            except (AttributeError, TypeError):  # no lower or can't call
                self._mapping[alt_form] = canon

        if self._case_insensitive:
            for prefix in self._canonical_prefixes:
                try:
                    self._mapping[prefix.lower()] = prefix
                except (AttributeError, TypeError):
                    pass

    def get_canonical_key(self, k):
        if k in self._canonical_prefixes:
            return k
        if k in self._mapping:
            return self._mapping[k]
        raise KeyError(k)

    def __getitem__(self, k):
        if k in self._canonical_prefixes:
            return self._canonical_prefixes[k]
        if k in self._mapping:
            return self._canonical_prefixes[self._mapping[k]]
        if self._case_insensitive:
            return self._canonical_prefixes[
                self._mapping[k.lower()]
            ]
        else:
            raise KeyError(k)

    def __len__(self) -> int:
        return len(self._canonical_prefixes)

    def __iter__(self):
        return iter(self._canonical_prefixes)


class CPDict(collections.abc.MutableMapping):
    """Case Preserving Dictionary"""
    def __init__(self, *args, **kwargs):
        self._cf_map: Dict[str, str] = {}
        self._canonical_dict: Dict[str, Any] = dict(*args, **kwargs)
        for key in self._canonical_dict:
            if not self.validate_key(key):
                raise ValueError(f"Invalid Key ({key})")
            self._cf_map[self.fold_key_case(key)] = key

    def __setitem__(self, k, v):
        if not self.validate_key(k):
            raise ValueError(f"Invalid Key ({k})")
        cf_key = self.fold_key_case(k)
        if cf_key in self._cf_map:
            canon_key = self._cf_map[cf_key]
            self._canonical_dict[canon_key] = v
        else:
            self._cf_map[cf_key] = k
            self._canonical_dict[k] = v

    def __delitem__(self, k):
        if not self.validate_key(k):
            raise KeyError(f"Invalid Key ({k})")
        canon_key = self._cf_map.pop(self.fold_key_case(k))
        del self._canonical_dict[canon_key]

    def __getitem__(self, k):
        if not self.validate_key(k):
            raise KeyError(f"Invalid Key ({k})")
        cf_key = self.fold_key_case(k)
        return self._canonical_dict[self._cf_map[cf_key]]

    def __len__(self) -> int:
        return len(self._canonical_dict)

    def __iter__(self) -> Iterator:
        return iter(self._canonical_dict)

    def get_canonical_key(self, key: str) -> str:
        return self._cf_map[self.fold_key_case(key)]

    @staticmethod
    def validate_key(key: Any) -> bool:
        if not isinstance(key, str):
            return False
        try:
            key.encode('ascii')  # only works for 7-bit ascii
            return True
        except UnicodeEncodeError:
            return False

    @staticmethod
    def fold_key_case(key: str) -> str:
        return key.lower()

    def __repr__(self):
        return f"CPDict: {repr(self._canonical_dict)}"

    def __str__(self):
        return f"CPDict: {str(self._canonical_dict)}"


class AgentsContainer:
    def __init__(self, cache_size: int = 128):
        self._agents: Dict[str, Tuple[str, str, float, IDLookupAgent]] = \
            CPDict()
        self.sources: Set[str] = set()
        self.targets: Set[str] = set()
        #: agents where src==tgt, used to verify id_value read from doc
        self.verify_agents: Dict[str, str] = {}
        self._frozen = False
        self.warn_frozen_msg = "Cannot alter agents when container is frozen"
        self.graph: Dict[str, Dict[str, List[str]]] = {}
        self.all_paths: Dict[Tuple[str, str],
                             Dict[Tuple[str, ...], float]] = {}
        self.prefixes = CPDict()
        for prefix in biolink_prefixes:
            if prefix == 'DBSNP':
                # TODO: Workaround for entries w/ diff. cases in biolink-model
                #  Fix biolink-model or resolve this!!!
                #  For whatever reason we have both DBSNP and dbSNP
                continue
            self.add_prefix(prefix)

    def add_prefix(self, prefix: str):
        if prefix in self.prefixes:
            raise ValueError(f"{prefix} already exists")
        if validate_prefix(prefix):
            self.prefixes[prefix] = prefix
        else:
            raise ValueError(f"{prefix} bad prefix syntax")

    def add(self, source: str, target: str, agent: IDLookupAgent,
            cost: float = 1.0, name: Optional[str] = None) -> str:
        if self._frozen:
            raise RuntimeError(self.warn_frozen_msg)
        canon_src = self.prefixes.get(source)
        if canon_src:
            source = canon_src
        else:
            warnings.warn(f"{source} is not a recognized prefix", Warning)
            # add to prefixes anyways after the warning, also shuts up
            # future warnings
            self.add_prefix(source)
        canon_tgt = self.prefixes.get(target)
        if canon_tgt:
            target = canon_tgt
        else:
            warnings.warn(f"{target} is not a recognized prefix", Warning)
            self.add_prefix(target)
        if name is None:
            name_base = f'{source}-{target}'
            if name_base not in self._agents:
                name = name_base
            else:
                idx = 1
                while True:
                    name = f'{name_base}_{idx}'
                    if name in self._agents:
                        idx += 1
                    else:
                        break
        self._agents[name] = (source, target, cost, agent)
        self.sources.add(source)
        self.targets.add(target)
        if source == target:
            self.verify_agents[target] = name
        return name

    def remove(self, name: str):
        if self._frozen:
            raise RuntimeError(self.warn_frozen_msg)
        del self._agents[name]

    def __getitem__(self, key) -> IDLookupAgent:
        return self._agents[key][3]

    @property
    def raw_agents(self):
        return self._agents

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, value):
        # FIXME: move the following line to a proper place
        self._build_all_paths()
        self._frozen = value

    def _construct_graph(self):
        self.graph = {}
        for agent_name, (src, dst, cost, _) in self._agents.items():
            if src not in self.graph:
                self.graph[src] = {}
            if src == dst:
                continue  # ignore those where src==dst
            self.graph[src].setdefault(dst, []).append(agent_name)

    def _generate_all_paths(self, current_node: str, path: List[str],
                            cost: float) \
            -> Generator[Tuple[Tuple[str, ...], str, float], None, None]:
        # TODO: total # of paths grows exponentially
        #  Add option to prevent loops (visiting the same node twice, even
        #  with different agents/edges) in path should fix it,
        #  at least for the foreseeable future
        if path:  # non empty path
            yield tuple(path), current_node, cost
        # if no more paths from here, leave
        if current_node not in self.graph:
            return None
        node = self.graph[current_node]
        for next_node, agents in node.items():
            # ignore loop back to same object (should not exist
            # in this graph anyways)
            if next_node == current_node:
                continue
            for agent in agents:
                # a same agent is not used twice
                if agent in path:
                    continue
                # don't visit a node twice
                past_nodes = set()
                for path_agent in path:
                    src, dst, _, _ = self._agents[path_agent]
                    past_nodes.update({src, dst})
                _, dst, _, _ = self._agents[agent]
                if dst in past_nodes:
                    continue
                new_path = path.copy()
                new_path.append(agent)
                nc = cost + self._agents[agent][2]
                yield from self._generate_all_paths(next_node, new_path, nc)

    def _build_all_paths(self):
        self._construct_graph()
        self.all_paths = {}
        for src in self.graph:
            for path, dst, cost in self._generate_all_paths(src, [], 0.):
                self.all_paths.setdefault((src, dst), {})[path] = cost

    def shortest_path_v2(self, src: str, dst: str,
                         not_start_with: Iterable[Iterable[str]]) -> \
            Tuple[Optional[Tuple[str, ...]], float]:
        paths = self.all_paths.get((src, dst), {})
        not_start_with = [tuple(x) for x in not_start_with]
        # TODO: maybe use a trie to speed up the not_start_with lookup
        # TODO: sort the paths dictionary beforehand (during construction)
        filtered_paths = {}
        for path, cost in paths.items():
            for reject_path in not_start_with:
                reject_len = len(reject_path)
                if len(path) < reject_len:
                    continue
                if path[:reject_len] == reject_path:
                    break
                continue
            else:  # did not break out of loop => no match in rejects
                filtered_paths[path] = cost
        if len(filtered_paths) > 0:
            shortest_path = min(filtered_paths, key=filtered_paths.get)
            ret = (shortest_path, filtered_paths[shortest_path])
        else:
            ret = None, float('inf')
        return ret


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

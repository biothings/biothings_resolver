import collections.abc
from typing import Any, Dict, Iterator, Callable, Optional, List, Tuple, \
    Mapping, Set, Iterable
from .agents import IDLookupAgent
from collections import OrderedDict, defaultdict


class IDPropertyContainer(collections.abc.MutableMapping):
    def __init__(self, key_normalizer: Optional[Callable[[str], str]] = None):
        self._dict = {}
        if key_normalizer is not None:
            self.key_normalizer = key_normalizer
        else:
            self.key_normalizer = lambda s: s

    def __delitem__(self, k) -> None:
        del self._dict[self.key_normalizer(k)]

    def __getitem__(self, k):
        return self._dict[self.key_normalizer(k)]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator:
        return iter(self._dict)

    def __setitem__(self, k, v) -> None:
        self._dict[self.key_normalizer(k)] = v


class AgentsContainer:
    def __init__(self, cache_size: int = 128):
        self._agents: Dict[str, Tuple[str, str, float, IDLookupAgent]] = {}
        self.sources: Set[str] = set()
        self.targets: Set[str] = set()
        #: agents where src==tgt, used to verify id_value read from doc
        self.verify_agents: Dict[str, str] = {}
        self.spt_cache = LRU(maxsize=cache_size)
        self.path_cache = LRU(maxsize=cache_size)
        self._frozen = False
        self.warn_frozen_msg = "Cannot alter agents when container is frozen"

    def add(self, source: str, target: str, agent: IDLookupAgent,
            cost: float = 1.0, name: Optional[str] = None) -> str:
        if self._frozen:
            raise RuntimeError(self.warn_frozen_msg)
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
        # invalidate all cache on status change
        self.spt_cache.clear()
        self.path_cache.clear()
        self._frozen = value

    def shortest_path_tree(self, target: str, reject_agents: Iterable[str]) ->\
            Tuple[Dict[str, List[str]], Dict[str, float]]:
        reject_agents = frozenset(reject_agents)
        cache_key = (target, reject_agents)
        if self._frozen and cache_key in self.spt_cache:
            return self.spt_cache[cache_key]
        # build a reversed directed graph
        graph: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        agent_costs = {}
        all_nodes = set()
        for agent_name in self._agents.keys() - reject_agents:
            a_src, a_tgt, a_cost, _ = self._agents[agent_name]
            graph.setdefault(a_tgt, {}).setdefault(
                a_src, []).append(agent_name)
            agent_costs[agent_name] = a_cost
            all_nodes.add(a_src)
            all_nodes.add(a_tgt)
        # sanity check
        if target not in all_nodes:
            out = ({}, {})
        else:
            # dijkstra
            spt_set = set()
            path_costs = {}
            path_dict = {target: []}
            for node_name in all_nodes:
                path_costs[node_name] = float('inf')
            path_costs[target] = 0.
            while spt_set != all_nodes:
                # pick nearest node not in spt_set
                curr_node = min(path_costs.keys() - spt_set,
                                key=path_costs.get)
                for src_node, agents in graph[curr_node].items():
                    # find agent with lowest cost
                    agent = min(agents, key=agent_costs.get)
                    agent_cost = agent_costs[agent]
                    # calculate total cost from src node to final target
                    new_cost = path_costs[curr_node] + agent_cost
                    # update
                    if path_costs[src_node] > new_cost:
                        path_costs[src_node] = new_cost
                        path_dict[src_node] = [agent] + path_dict[curr_node]
                # visited, so add to spt_set
                spt_set.add(curr_node)
            # done
            # remove target node, it will be empty/0. anyways
            path_dict.pop(target)
            path_costs.pop(target)
            out = (path_dict, path_costs)
        # check if needs updating cache
        if self._frozen:
            self.spt_cache[cache_key] = out
        return out

    def shortest_path(self, sources: Iterable[str], target: str,
                      reject_agents: Iterable[str]) -> Optional[List[str]]:
        sources = frozenset(sources)
        reject_agents = frozenset(reject_agents)
        cache_key = (target, sources, reject_agents)
        if self._frozen and cache_key in self.path_cache:
            return self.path_cache[cache_key]

        if target in sources:
            out = None
        else:
            paths, costs = self.shortest_path_tree(target, reject_agents)
            common_sources = set(sources) & paths.keys()
            if len(common_sources) == 0:
                out = None
            else:
                source = min(set(sources) & costs.keys(), key=costs.get)
                out = paths[source]
        if self._frozen:
            self.path_cache[cache_key] = out
        return out


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


class IDStructure(collections.abc.MutableMapping):
    def __init__(self, mapping: Optional[Mapping] = None):
        super(IDStructure, self).__init__()
        self._done = False
        self.tracing = False
        self.trace: List[Tuple[str, Any]] = []
        if mapping:
            self._data = dict(mapping)
        else:
            self._data = {}

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v
        if self.tracing:
            self.trace.append((f'setitem_{k}', v))

    def __delitem__(self, k):
        del self._data[k]
        if self.tracing:
            self.trace.append((f'delitem_{k}', ''))

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __str__(self):
        kv_pair_str = ', '.join([f'{k}={v}' for k, v in self._data.items()])
        return 'IDStructure(' + kv_pair_str + ')'

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, v: bool):
        if v and not self._done:
            self._done = True
            if self.tracing:
                self.trace.append(('set_done', 'True'))
        else:
            raise ValueError("setting done to False or setting True twice")

    def set_id_value(self, id_type: str, id_value: str,
                     agent_name: Optional[str] = None,
                     trace_value: Any = None):
        if self.tracing and agent_name:
            self._data[id_type] = id_value
            if trace_value:
                self.trace.append((agent_name, trace_value))
            else:
                self.trace.append((agent_name, id_value))
        else:
            self[id_type] = id_value

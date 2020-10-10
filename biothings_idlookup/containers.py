import collections.abc
from enum import Enum
from typing import Any, Dict, Iterator, Callable, Optional, List, Tuple, \
    Mapping, Set, Iterable, Generator
from collections import OrderedDict, defaultdict

from .agents import IDLookupAgent


class AgentsContainer:
    def __init__(self, cache_size: int = 128):
        self._agents: Dict[str, Tuple[str, str, float, IDLookupAgent]] = {}
        self.sources: Set[str] = set()
        self.targets: Set[str] = set()
        #: agents where src==tgt, used to verify id_value read from doc
        self.verify_agents: Dict[str, str] = {}
        self._frozen = False
        self.warn_frozen_msg = "Cannot alter agents when container is frozen"
        self.graph: Dict[str, Dict[str, List[str]]] = {}
        self.all_paths: Dict[Tuple[str, str],
                             Dict[Tuple[str, ...], float]] = {}

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

# -*- coding: utf-8 -*-
""" Resolver for identifier types of biological entities

Resolver is used to easily cast between various identifier types that
a biological entity may have. Instead of directly querying an API like
BioThings or searching for cross-references in a database, Resolver
provides a unified interface for looking up identifiers in batches. It
is not only capable of direct queries, it can also perform indirect
queries: when there is not a direct path an identifier of one type is
first translated to one or more identifiers of intermediate types,
before finally obtaining the target identifier type.

The `Resolver` class implements the lookup mechanisms. The `Agent` class
performs the lookups.
One part of the core mechanism is implemented in the `Resolver` class,
which runs through va
which tries to perform the identifier lookup in as little steps as possible, using pathfinding algorithms.
This functionality can be accessed using the IDLookup.lookup_identity method.

It also performs obtaining information from documents, as well as
performing transforms on input and output identifiers and such,
to facilitate looking up and updating identifiers in documents.

Identity Translation Agents: subclass of IDLookupAgent. Agents perform lookup on a batch of identities.
This is used to be called an Edge, but due to 1) this may cause confusion in the topological sort part,
because agents (edges) are nodes in the DAG and 2) we are hiding from the user the concept of graphs,
so (I think) it is reasonable to call them Identity Translation Agents instead of Edges.

To use an agent, it has to be registered in an `IDLookup` instance.
"""

import copy
import itertools
import logging
from functools import wraps
from typing import List, Dict, Optional, Tuple, Union, Callable, Iterable, \
    Generator, Sequence, Mapping, Set, MutableMapping

from .containers import AgentsContainer, CPDict
from .utils import dict_get_nested, transform_str
from .curie import split_curie


class Resolver:
    """Base class for looking up IDs

    Attributes:
        preferred: List of output id_types in the order of preference
        batch_size: Size of documents to process in each batch

    """
    def __init__(self, input_types=None):
        super(Resolver, self).__init__()
        self.logger = logging.getLogger('biothings_resolver')
        self.preferred: List[str] = []
        self.in_fields: Dict[str, Union[str, Callable[[dict], str]]] = {}
        self.agents = AgentsContainer(cache_size=128)
        self.batch_size = 1000
        # now we store failed processors on a per id basis
        self.id_failed_agents = {}

        self.id_v_in_xfrm: \
            MutableMapping[str, List[Callable[[str], Optional[str]]]] = CPDict()
        self.id_v_out_xfrm: \
            MutableMapping[str, List[Callable[[str], Optional[str]]]] = CPDict()

        self.curie_in_xfrm: \
            MutableMapping[str, Callable[[str], Optional[str]]] = CPDict()
        self.curie_out_xfrm: \
            MutableMapping[str, Callable[[str], Optional[str]]] = CPDict()

        # (id_t, id_v): List[(prev. id_t, id_v, agt)]
        self.resolver_trace: Dict[Tuple[str, str], Set[tuple]] = {}

        self.document_resolve_id_field = '_id'

        if input_types:
            for input_type, input_field in input_types:
                self.add_input_field(input_type, input_field)

    def add_input_field(self, id_type: str, field: Union[str, Callable[[dict], str]]):
        """Register an input field

        Args:
            id_type: identifier type of this field
            field: Either a `str`, for dot delimited notation for the key in the document; OR
                a `Callable` that takes in the entire document and returns a `str`.
        """
        self.in_fields[id_type] = field

    def remove_input_field(self, id_type: str):
        """Remove a input field"""
        del self.in_fields[id_type]

    def add_id_transforms(self, direction: str, id_type: str, normalizer: Callable[[str], Optional[str]]) -> None:
        """Add id_value transformations on input/output, when transforming documents

        Args:
            direction: 'input' or 'output'
            id_type: type of identity to apply transform
            normalizer: transformation, return None if an id_value is invalid and needs to be removed

        Examples:
            For instance, if a document has may or may not have InChI ids, and pandas reading the document returns NaN,
            it is preferred to have them removed because they are invalid input and slows down lookup.
            >>> lookup = Resolver()
            >>> lookup.add_id_transforms('input', 'inchi', lambda s: None if s != s else s)
            An input can also be skipped based on regex matching, see biothings_resolver.transforms.filter_regex

            For instance, if the output of an PubChem identity needs to be prefixed with 'CID'
            >>> lookup = Resolver()
            >>> lookup.add_id_transforms('output', 'pubchem', lambda s: f"CID {s}")

            Some common uses are implemented in biothings_resolver.transforms and can be used as examples
            on implementing new transforms
        """
        # TODO: implement transforms that applies to ALL identity types (old behavior)
        if direction == 'input':
            self.id_v_in_xfrm.setdefault(id_type, []).append(normalizer)
        elif direction == 'output':
            self.id_v_out_xfrm.setdefault(id_type, []).append(normalizer)
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def __call__(self, func):
        """Decorates a data_loading function

        Args:
            func: function to be decorated

        Returns:
            New function that has an updated `_id` field
        """
        self.logger.debug("decorating function %s", func)

        @wraps(func)
        def wrapped_f(*args, **kwargs):
            input_docs = func(*args, **kwargs)
            while True:
                chunk_docs = list(itertools.islice(input_docs, self.batch_size))
                num_docs = len(chunk_docs)
                if num_docs == 0:
                    self.logger.debug("no more documents, returning.")
                    break
                self.logger.debug("got %d documents input in this batch", num_docs)
                yield from self.resolve_document(chunk_docs)
            return None
        return wrapped_f

    def resolve_identifier(self, ids: Sequence[Mapping[str, str]],
                           expand: bool) -> \
            Generator[Dict[str, List[str]], None, None]:
        input_ids: List[Dict[str, List[str]]] = []
        # convert input
        for id_struct in ids:
            d = {}
            for id_t, id_v in id_struct.items():
                # FIXME: MOVE to an appropriate place
                # canonicalize id type
                id_t = self.agents.prefixes.get_canon_key(id_t)
                if not id_t:
                    continue
                # convert each id to list of str
                if isinstance(id_v, list):
                    d[id_t] = [str(x) for x in id_v]
                else:
                    d[id_t] = [str(id_v)]
            input_ids.append(d)
        while True:
            # build lookup path for each id
            resolve_q = self._build_resolve_queue(input_ids, expand)
            if len(resolve_q) == 0:
                break  # exit loop when nothing to do
            agent_resolve_id_info = {}  # what id should each agent lookup
            paths = []  # "linked" paths, one agent points to the next
            id_path = [-1] * len(input_ids)
            for path_idx, (path, items) in enumerate(resolve_q.items()):
                # populate the items for the first agent in path
                agent_resolve_id_info.setdefault(path[0], []).extend(items)
                path_link = {}
                for a1_name, a2_name in zip(path[:-1], path[1:]):
                    path_link[a1_name] = a2_name
                paths.append(path_link)
                for ids_idx, _, _ in items:
                    id_path[ids_idx] = path_idx
            lookup_order = self._topological_sort(resolve_q.keys())
            for agent_name in lookup_order:
                src_t, tgt_t, _, agent = self.agents.raw_agents[agent_name]
                src_values = {}  # map values back to original indices
                for idx_info in agent_resolve_id_info.get(agent_name, []):
                    ids_idx, id_t, idv_idx = idx_info
                    src_idv = input_ids[ids_idx][id_t][idv_idx]  # id_value
                    src_values.setdefault(src_idv, []).append(ids_idx)
                if len(src_values.keys()) == 0:
                    continue  # in the case nothing is left
                results = agent.lookup(src_values.keys())
                for src_idv, indices in src_values.items():
                    result = results.get(src_idv, None)
                    if result is None:
                        self.logger.info("%s did not process %s:%s",
                                         agent_name, src_t, src_idv)
                        for (id_t, id_v), failed_path in self._build_fail_path(
                                (src_t, src_idv), [agent_name]
                        ):
                            self.id_failed_agents.setdefault(
                                (id_t, id_v), set()).add(failed_path)
                            self.logger.debug(
                                "adding reject path %s to (%s, %s)",
                                failed_path, id_t, id_v
                            )
                        continue
                    if isinstance(result, list):
                        result = [str(r) for r in result]
                    else:
                        result = [str(result)]
                    self.logger.debug("got %s:%s -> %s:%s", src_t,
                                      src_idv, tgt_t, result)
                    # update trace
                    for r in result:
                        self.resolver_trace.setdefault(
                            (tgt_t, r), set()
                        ).add((src_t, src_idv, agent_name))
                    for ids_idx in indices:
                        id_l = input_ids[ids_idx].setdefault(tgt_t, [])
                        orig_len = len(id_l)
                        # FIXME: deal with dupes
                        id_l.extend(result)
                        curr_len = len(id_l)
                        path_id = id_path[ids_idx]
                        assert path_id != -1  # -1 means no path assigned
                        next_agent = paths[path_id].get(agent_name, None)
                        if next_agent:
                            agent_resolve_id_info.setdefault(
                                next_agent, []
                            ).extend([(ids_idx, tgt_t, v)
                                      for v in range(orig_len, curr_len)])
                    self.logger.debug("updated %s", indices)
            # end of while loop
        for id_struct in input_ids:
            od = {}
            for id_t in self.preferred:
                if id_t in id_struct:
                    # FIXME: same as above
                    od[id_t] = list(set(id_struct[id_t]))
                    if expand:
                        continue
                    else:
                        break
            yield od

    # recursively build failure trace
    def _build_fail_path(self, prev_id: Tuple[str, str], path: list):
        yield prev_id, tuple(reversed(path))
        if prev_id not in self.resolver_trace:
            return
        for id_t, id_v, a_name in self.resolver_trace[prev_id]:
            if a_name in path:
                continue
            new_path = path.copy()
            new_path.append(a_name)
            yield from self._build_fail_path((id_t, id_v), new_path)

    @staticmethod
    def _topological_sort(tasks: Iterable[Iterable[str]]) -> List[str]:
        # build DAG
        dep_graph = {}
        for path in tasks:
            # not using pairwise, always want the first one
            path = list(path)[::-1]  # reversed
            path_length = len(path)
            for task_idx, task_name in enumerate(path):
                prerequisites = dep_graph.setdefault(task_name, set())
                if task_idx + 1 < path_length:
                    prerequisites.add(path[task_idx + 1])
        # sort DAG
        tasks_order = []
        zero_indegree = [t_name for t_name, deps in
                         dep_graph.items() if len(deps) == 0]
        while zero_indegree:
            task_name = zero_indegree.pop()
            tasks_order.append(task_name)
            del dep_graph[task_name]
            for other_agent, deps in dep_graph.items():
                if task_name in deps:
                    deps.remove(task_name)
                    if len(deps) == 0:
                        zero_indegree.append(other_agent)
        if dep_graph:
            # FIXME: we can break cycles (just requires extra lookup)
            raise RuntimeError("Loop in dependency graph for agents")
        return tasks_order

    def _build_resolve_queue(self, input_ids, resolve_all):
        resolve_queue = {}  # path: starting point tuple
        for id_idx, id_struct in enumerate(input_ids):
            # anything we already have has initial cost of 0
            # hence why we store them and use them as available sources
            # we store/retrieve failed agents per "starting point"
            for target in self.preferred:
                if target in id_struct:
                    if resolve_all:
                        continue
                    else:
                        break
                possible_paths = {}
                for id_t, id_vs in id_struct.items():
                    for idv_idx, idv in enumerate(id_vs):
                        rej_starts = self.id_failed_agents.get((id_t, idv), [])
                        path, cost = self.agents.shortest_path_v2(
                            id_t, target, rej_starts
                        )
                        if path is not None:
                            possible_paths[(id_t, idv_idx, path)] = cost
                if len(possible_paths) == 0:
                    continue
                id_t, idv_idx, path = min(possible_paths,
                                          key=possible_paths.get)
                if path is not None:
                    resolve_queue.setdefault(path, []).append(
                        (id_idx, id_t, idv_idx)
                    )
                    break  # found a valid path
            else:  # no valid path found
                pass
        return resolve_queue

    def resolve_curie(self, curies: Sequence[str], expand: bool) -> \
            Generator[List[str], None, None]:
        id_l = []
        for curie in curies:
            prefix, ref = split_curie(curie)
            xfrm = self.curie_in_xfrm.get(prefix)
            if xfrm:
                id_v = self.curie_in_xfrm[prefix](curie)
            else:
                id_v = ref
            id_l.append({prefix: id_v})
        for id_resolved in self.resolve_identifier(id_l, expand):
            output = []
            for prefix, id_values in id_resolved.items():
                xfrm = self.curie_out_xfrm.get(prefix)
                for id_v in id_values:
                    if xfrm:
                        id_v = xfrm(id_v)
                    else:
                        pass  # no transform
                    output.append(f"{prefix}:{id_v}")
            yield output

    def resolve_document(self, documents: Iterable[dict]) -> Generator:
        """Resolve identifiers given a document

        Args:
            documents: a series of documents

        Returns:
            List of documents, with duplications when one set of
            identifiers gives more than one output of the desired
            identifier type.
        """
        id_dicts = []
        o_docs = []
        for document in documents:
            o_docs.append(copy.deepcopy(document))
            id_dict = {}  # {id_type: id_values}
            for id_type, field in self.in_fields.items():
                if isinstance(field, str):  # obtain value from single field
                    id_v = dict_get_nested(document, field)
                elif isinstance(field, Callable):
                    # obtain value using custom implementation given
                    id_v = field(document)
                else:
                    raise RuntimeError("unrecognized field type")
                if isinstance(id_v, str):
                    pass  # do nothing when we have str
                elif id_v is None:
                    continue  # skip when None is obtained
                else:
                    pass
                xfrm = self.id_v_in_xfrm.get(id_type, [])
                id_v = transform_str(id_v, xfrm)
                if id_v is not None:
                    if not isinstance(id_v, str):
                        self.logger.warning(
                            "got raw %s:(%s)%r and force converting to str...",
                            id_type, type(id_v).__name__, id_v)
                        id_v = str(id_v)
                        self.logger.warning("...converted to %s", id_v)
                    id_dict[id_type] = id_v
            # end of extracting for all fields in single document
            id_dicts.append(id_dict)
        # end of extracting IDs from documents
        for od, res in zip(o_docs, self.resolve_identifier(id_dicts, False)):
            o_idv = od.get(self.document_resolve_id_field, None)
            for id_t in self.preferred:
                if id_t in res:
                    if len(res[id_t]) > 1:
                        self.logger.warning("%s->%s(multi)", o_idv, res[id_t])
                    for id_v in res[id_t]:
                        xfrm = self.id_v_out_xfrm.get(id_t, [])
                        id_v = transform_str(id_v, xfrm)
                        new_doc = copy.deepcopy(od)
                        new_doc[self.document_resolve_id_field] = id_v
                        self.logger.debug("updated doc ID %s->%s", o_idv, id_v)
                        yield new_doc
                    break
            else:  # did not break from loop == no new id
                self.logger.debug("did not update %s", o_idv)
                yield od  # no need to copy

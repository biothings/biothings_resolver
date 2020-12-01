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
    Generator, Sequence, Mapping, Set, MutableMapping, Any, Collection

from .containers import AgentsContainer, CPDict, CanonDict
from .utils import dict_get_nested, transform_str
from .curie import split_curie


class Resolver:
    """Base class for looking up IDs

    Attributes:
        preferred (List[str]): A list of preferred output types, in the order
            of preference, like CURIE-prefix style.
        batch_size (int):  The maximum number of items that a resolving agent
            will be asked to handle at once. An agent may be asked to process
            less number of items than `batch_size` at one time, but will never
            be asked to process more items.
        expand (bool):  Whether any resolving functions will attempt to return
            all the preferred output types, or only return the available most
            preferred output type.
        in_xfrm (MutableMapping[str, Callable[[str, Any], Any]): Input
            transformations. This takes place before resolving through resolver
            agents.

            Keys are CURIE-prefix style or '*', values are functions that take
            the CURIE-prefix as the first argument, and the value as the second
            argument. If the return value is not `None` it will be used to
            update the value, or else the input is discarded.

            When the key '*' exists, it will be applied to all input types,
            after any other transformations.

            Examples:
                For instance to remove the 'CID' prefix for PubChem CID inputs,
                we can have all 'PUBCHEM.COMPOUND' have a transformation that
                strips the first three characters of the input:

                >>> resolver = Resolver()
                >>> resolver.in_xfrm['PUBCHEM.COMPOUND'] = lambda t, v: v[3:]

        out_xfrm (MutableMapping[str, Callable[[str, Any], Any]): Output
            transformations. This takes place after resolving through resolver
            agents. Key and values are the same specifications as in
            :py:attr:`in_xfrm`.

            Notes:
                - To perform multiple transforms, compose a function that
                  performs multiple transforms.
                - To apply transformations between resolver agents, customize
                  the agents.

    """
    def __init__(self, input_types=None):
        super(Resolver, self).__init__()
        self.logger = logging.getLogger('biothings_resolver')
        self.logger.setLevel(logging.CRITICAL)
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

        self.in_xfrm: MutableMapping[str, Callable] = CanonDict()
        self.out_xfrm: MutableMapping[str, Callable] = CanonDict()

        # (id_t, id_v): List[(prev. id_t, id_v, agt)]
        self.resolver_trace: Dict[Tuple[str, str], Set[tuple]] = {}

        self.document_resolve_id_field = '_id'

        self.expand = False

        if input_types:
            for input_type, input_field in input_types:
                self.add_input_field(input_type, input_field)

        self.decorators = Decorators(self)

        self._debug = False

        self._opt_cnf = {
            'vt_key': '\x00\x00\x00\x00*origin_vt',  # key for value types
        }

    @property
    def debug(self) -> bool:
        """bool:Debug setting

        Controls whether debugging information will be output using the
        standard logging utilities. It will be the same as setting up logging
        manually, this is only a convenience feature.

        Notes:
            To observe how a single input is resolved, set :py:attr:`debug` to
            `True` and invoke any resolving functions with a single input.

        """
        return self._debug

    @debug.setter
    def debug(self, v: bool):
        if v:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.CRITICAL)
        self._debug = v

    @property
    def max_path_length(self) -> int:
        """int:Maximum number of agents to go through during resolve.
        """
        return self.agents.max_path

    @max_path_length.setter
    def max_path_length(self, length: int) -> None:
        self.agents.frozen = False
        self.agents.max_path = length
        self.agents.frozen = True

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

    def resolve(self, in_values: Collection[Dict[str, Any]]) -> \
            Generator[Dict[str, list], None, None]:
        """Resolve values

        Takes in an ordered collection (for instance, a `list`) of
        dictionaries. Each dictionary represents a group of inputs, combined
        together in a similar fashion to a logical OR operation, which means
        an input of any type in the group will be used for resolving to the
        desired output types. The keys of the input dictionary is a
        CURIE-prefix for the corresponding type.

        The output is in the same order as the input. The dictionary will have
        keys specified in `Resolver.preferred`, instead of the canonical
        version. For instance, if "fb" is an alias to the canonical prefix
        "FLYBASE", and "fb" is specified in the :py:attr:`preferred`
        attribute,
        then the output dictionary will use "fb" as its key when an output
        value is of type "FLYBASE". If any results are produced for a type of
        output value, it will be stored in the `list`, which has at least one
        element (i.e., no key if no result, singular results will not be taken
        out of `list` container). If a group of input values does not produce
        an output, the output will be an empty dictionary.

        Only the resolution results will be in the output (as configured in
        :py:attr:`preferred` and :py:attr:`expand`). To copy other
        fields from the input, use the :py:meth:`resolve_document` method
        below.


        Args:
            in_values: Ordered collection of dicts, where keys are CURIE-prefix
                style str indicators of value types.
        Yields:
            dicts where keys are CURIE-prefix style and values are list of
                resolved values.
        """
        it = iter(in_values)
        while True:
            chunk_input = list(itertools.islice(it, self.batch_size))
            num_docs = len(chunk_input)
            if num_docs == 0:
                break
            # canonicalize not needed here, the method below handles that
            yield from self.resolve_identifier(chunk_input)
        return None

    def resolve_identifier(self, ids: Sequence[Mapping[str, str]]) -> \
            Generator[Dict[str, List[str]], None, None]:
        input_ids: List[Dict[str, List[str]]] = []
        # convert input
        vt_key = self._opt_cnf['vt_key']
        for id_struct in ids:
            d = {
                vt_key: []
            }
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
                d[vt_key].append(id_t)
            input_ids.append(d)
        while True:
            # build lookup path for each id
            resolve_q = self._build_resolve_queue(input_ids)
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
                    id_t_canon = self.agents.prefixes.get_canon_key(id_t)
                    od[id_t] = list(set(id_struct[id_t_canon]))
                    if self.expand:
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

    def _build_resolve_queue(self, input_ids):
        resolve_queue = {}  # path: starting point tuple
        for id_idx, id_struct in enumerate(input_ids):
            # anything we already have has initial cost of 0
            # hence why we store them and use them as available sources
            # we store/retrieve failed agents per "starting point"
            for target in self.preferred:
                if target in id_struct:
                    if self.expand:
                        continue
                    else:
                        break
                possible_paths = {}
                for id_t in id_struct[self._opt_cnf['vt_key']]:
                    id_vs = id_struct.get(id_t)
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

    def resolve_curie(self, curies: Sequence[str]) -> \
            Generator[List[str], None, None]:
        """Resolve CURIE style input

        Takes in an ordered collection of CURIEs. Each CURIE is an input.

        Produces `list`s of `str`, in the same order as the input. The CURIE
        prefix in the output will use the version specified in
        `Resolver.preferred` instead of the canonical version. If an input has
        multiple output values of the same type, they will all be present in
        the output `list`. If an input does not produce an output, its
        corresponding output will be an empty `list`.

        Args:
            curies: Ordered collection of CURIE style inputs

        Yields:
            List of CURIE-style results
        """
        id_l = []
        for curie in curies:
            prefix, ref = split_curie(curie)
            xfrm = self.curie_in_xfrm.get(prefix)
            if xfrm:
                id_v = self.curie_in_xfrm[prefix](curie)
            else:
                id_v = ref
            id_l.append({prefix: id_v})
        for id_resolved in self.resolve_identifier(id_l):
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
        for od, res in zip(o_docs, self.resolve_identifier(id_dicts)):
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


class Decorators:
    def __init__(self, parent):
        super(Decorators, self).__init__()
        self.parent = parent

    def resolve(self, func):
        func_name = 'resolve'
        if not hasattr(self.parent, func_name):
            raise AttributeError(f"Parent does not have {func_name}")

        @wraps(func)
        def wrapped(*args, **kwargs):
            input_docs = func(*args, **kwargs)
            yield from self.parent.resolve(input_docs)

        return wrapped

    def resolve_curie(self, func):
        func_name = 'resolve_curie'
        if not hasattr(self.parent, func_name):
            raise AttributeError(f"Parent does not have {func_name}")

        @wraps(func)
        def wrapped(*args, **kwargs):
            input_docs = func(*args, **kwargs)
            yield from self.parent.resolve_curie(input_docs)

        return wrapped

    def resolve_document(self, func, *a, **kwa):
        func_name = 'resolve_document'
        if not hasattr(self.parent, func_name):
            raise AttributeError(f"Parent does not have {func_name}")

        @wraps(func)
        def wrapped(*args, **kwargs):
            input_docs = func(*args, **kwargs)
            yield from self.parent.resolve_document(input_docs, *a, **kwa)

        return wrapped

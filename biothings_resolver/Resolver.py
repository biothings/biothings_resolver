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

The :py:class:`Resolver` class implements the lookup mechanisms. The
:py:class:`IDLookupAgent` class
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
import logging
from collections import OrderedDict
from functools import wraps
from typing import List, Dict, Optional, Tuple, Union, Callable, Iterable, \
    Generator, Sequence, Mapping, Set, Any, Collection
from uuid import UUID, uuid4

from .agents import ResolverAgent
from .curie import split_curie
from .utils import dict_get_nested


# define data types for performance and code readability
# some say __slots__ magic more performant than namedtuple
class _ValueRecord:
    __slots__ = ('value', 'type', 'origin')

    def __init__(self, value: Any, value_type: str,
                 origin: Optional[Tuple[UUID, UUID]] = None):
        self.value = value
        self.type: str = value_type
        self.origin: Optional[Tuple[UUID, UUID]] = origin
        # origin: (source_value, agent)


class _BufferObj:
    __slots__ = ('_data', 'failed', 'type', 'done', 'paths', 'resolved')

    def __init__(self):
        self._data: Dict[UUID, _ValueRecord] = dict()
        self.failed: Dict[UUID, Dict[UUID, Set[str]]] = dict()
        self.type: Dict[str, Set[UUID]] = dict()
        self.paths: Dict[Tuple[UUID, UUID], List[UUID]] = dict()
        self.done: bool = False
        self.resolved: Dict[UUID, Dict[str, Dict[UUID, Set[UUID]]]] = dict()
        # source_uuid, output_type, agent_uuid, result_uuids

    def __getitem__(self, item) -> _ValueRecord:
        return self._data[item]

    @property
    def data(self):  # I would prefer if there's a way to get a read-only view
        return self._data

    def add_value(self, value: _ValueRecord) -> UUID:
        u = uuid4()  # I don't see how this needs to be distributed
        self._data[u] = value
        self.type.setdefault(value.type, set()).add(u)
        if value.origin:
            source, agent = value.origin
            self.resolved.setdefault(
                source, dict()
            ).setdefault(
                value.type, dict()
            ).setdefault(
                agent, set()
            ).add(u)
        return u

    def add_path(self, start: UUID, path: List[UUID],
                 agents: Dict[UUID, ResolverAgent]):
        """
        Add a path(s) into object

        Handles intermediate resolved results

        Args:
            path: list of reversed path (first step is the last element)
        """
        path = path.copy()
        agent = path.pop()
        if len(path) > 0:
            next_agent = path[-1]
            next_type = agents[next_agent].input_type
            try:
                results = self.resolved[start][next_type][agent]
                for intermediate_result in results:
                    self.add_path(intermediate_result, path, agents)
                    return None
            except KeyError:
                pass
        # only when this is the only step, or the first non-resolved step
        # we add it to paths
        # - failed agents won't be added because it's eliminated during
        #   path finding
        # - one step resolve won't be resolved twice because it's the last step
        #   so either it has failed or got a result
        self.paths[(start, agent)] = path


class _PathInfo:
    __slots__ = ('agents', 'start', 'cost')
    def __init__(self, start: UUID, agents: List[UUID],
                 cost: float = float('inf')):
        self.start = start
        self.agents = agents.copy()
        self.cost = cost


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

        decorators: Container for decorators. :py:meth:`Decorators.resolve`,
            :py:meth:`Decorators.resolve_curie`, and
            :py:meth:`Decorators.resolve_document` exist and wraps a function
            inside the corresponding :py:meth:`resolve`,
            :py:meth:`resolve_curie`, and :py:meth:`resolve_document` in this
            class :py:class:`Resolver`.

    """
    def __init__(self):
        super(Resolver, self).__init__()
        self.logger = logging.getLogger('biothings_resolver')

        self.agents: Dict[UUID, ResolverAgent] = {}
        self.agent_costs: Dict[Tuple[UUID, str], float] = {}
        self.preferred: List[str] = []
        self.expand = False

        self.multiple_input_operation = 'any'  # rename, intersection, any

        self.max_buffer_size = 5000
        self.max_batch_size = 1000
        self.max_path_length = 3

        # TODO: xfrms

    @property
    def _agents_from_src(self) -> Dict[str, Set[UUID]]:
        flat_graph = {}
        for agent_uuid, agent in self.agents.items():
            i_type = agent.input_type
            flat_graph.setdefault(i_type, set()).add(agent_uuid)
        return flat_graph

    def _compute_paths_for_obj(self, obj: _BufferObj):
        # resolve step should not care if a type already exist in input
        potential_paths = {}
        for u, value in obj.data.items():
            wanted_types = set(self.preferred) - set(obj.resolved[u].keys())
            if value.origin is not None:
                continue  # ignore for non-input (i.e. intermediate) values
            failed = obj.failed[u]
            path_info = self.resolve_path(value.type, wanted_types,
                                          frozenset(failed))
            for dst_type, path_cost in path_info.items():
                potential_paths.setdefault(dst_type, dict())[u] = path_cost
        # let's forget about expand and union/intersect for now
        # FIXME: implement those parts later
        if self.expand or self.multiple_input_operation != 'any':
            raise NotImplementedError
        if not self.expand:
            for pref_type in self.preferred:
                if pref_type in potential_paths:
                    paths = potential_paths[pref_type]
                    best_path_uuid = sorted(paths, key=lambda x: x[1])[0]
                    path = list(reversed(paths[best_path_uuid]))
                    obj.add_path(best_path_uuid, path, self.agents)
                    return
            obj.done = True

        # deal with expand/ (any/union/intersect)
        # in case of expand=F, op=any, we should stop as soon as we have
        # one of the resolved

    def resolve(self, input_objects: Iterable[Mapping[str, Iterable]]) -> \
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
            input_objects: Ordered collection of dicts, where keys are
            CURIE-prefix style str indicators of value types.
        Yields:
            dicts where keys are CURIE-prefix style and values are list of
                resolved values.
        """
        buffer = OrderedDict()
        per_agent_queue: Dict[UUID, List[Tuple[UUID, UUID]]] = {}
        for in_obj in input_objects:
            # we are going to need per input-value level
            # failure info
            u = uuid4()
            o = _BufferObj()
            for vt, values in in_obj.items():
                for value in values:
                    d = _ValueRecord(copy.deepcopy(value), vt)
                    o.add_value(d)
            self._compute_paths_for_obj(o)
            for start, agent in o.paths:
                per_agent_queue.setdefault(agent, list()).append((u, start))
            buffer[u] = o
            verified_no_queue_full = False
            while not verified_no_queue_full:
                for agent_uuid, queue in per_agent_queue.items():
                    if len(queue) > min(self.max_batch_size,
                                        self.agents[agent_uuid].max_batch_size):
                        self._handle_agent_queue(agent_uuid, queue, buffer)
                        break
                else:  # no breaks occur == no len(queue) > max size
                    verified_no_queue_full = True
            if len(buffer) >= self.max_buffer_size:
                self._handle_buffer(buffer)  # try to yield first item
                self._yield_buffer(buffer)
        # clear the buffer after no more inputs
        while len(buffer) >= 0:
            self._handle_buffer(buffer)
            yield from self._yield_buffer(buffer)

    def _handle_agent_queue(self, agent_uuid: UUID,
                            queues: Dict[UUID, List[Tuple[UUID, UUID]]],
                            buffer: Mapping[UUID, _BufferObj]):
        queue = queues.get(agent_uuid, [])
        agent = self.agents[agent_uuid]
        resolve_types = agent.output_types
        input_values = []
        input_ids = []
        for ids in queue:
            bufobj_id, value_id = ids
            value = buffer[bufobj_id].data[value_id].value
            input_values.append(value)
            input_ids.append(ids)
        results = agent.lookup(input_values, resolve_types)
        for (bufobj_id, value_id), result in zip(input_ids, results):
            bufobj = buffer[bufobj_id]
            # pop the path. If we will add later if new_value has what we need
            path = bufobj.paths.pop((value_id, agent_uuid))
            try:
                n_agent_id = path.pop()
                nxt_type = self.agents[n_agent_id].input_type
            except IndexError:
                n_agent_id = None
                nxt_type = None
            failed = set(resolve_types) - result.keys()
            # update successful ones
            for r_type, r_values in result.items():
                for r_value in r_values:
                    vr = _ValueRecord(r_value, r_type, (value_id, agent_uuid))
                    n_value_id = bufobj.add_value(vr)
                    # handle path
                    if r_type == nxt_type:
                        # new values certainly not cached
                        # path is certainly not zero length
                        bufobj.paths[(n_value_id, n_agent_id)] = path.copy()
                        queues.setdefault(n_agent_id, list()).append(
                            (bufobj_id, n_value_id)
                        )
            # update failed ones
            for f_type in failed:
                bufobj.failed.setdefault(value_id, dict()).setdefault(
                    agent_uuid, set()
                ).add(f_type)

    def _handle_buffer(self, buffer):
        """
        Do whatever to mark the first item as done
        """
        pass

    def _yield_buffer(self, buffer: OrderedDict[UUID, _BufferObj]):
        del_items = []
        for obj_id, obj in buffer.items():
            if not obj.done:
                break
            copy = True
            output = {}
            for o_type in self.preferred:
                v_ids = obj.type.get(o_type, set())
                o_list = []
                for v_id in v_ids:
                    vr = obj.data[v_id]
                    assert vr.type == o_type
                    if copy is True:
                        o_list.append(vr.value)
                    elif vr.origin is not None:
                        o_list.append(vr.value)
                if o_list:  # non-empty
                    output[o_type] = o_list
                    if not self.expand:
                        break
            yield output
            del_items.append(obj_id)
        for obj_id in del_items:
            del buffer[obj_id]

    def resolve_path(self, src: str, dsts: Collection[str],
                     failed_combos: frozenset) \
            -> Dict[str, Tuple[Tuple[UUID, ...], float]]:
        # I think we can call this BFS with pruning? we start from src
        # and visit everything, but only keeping the most economical
        flat_graph = self._agents_from_src
        paths = {src: ([], 0.)}
        for path_length in range(self.max_path_length):
            srcs = []
            for src, (path, running_cost) in paths.items():
                if len(path) != path_length:
                    continue  # only do new ones
                srcs.append(src)
            while len(srcs) > 0:
                src = srcs.pop()
                path, running_cost = paths[src]
                if src not in flat_graph:
                    continue
                for agent_uuid in flat_graph[src]:
                    agent = self.agents[agent_uuid]
                    for o_type in agent.output_types:
                        if (agent_uuid, o_type) in failed_combos:
                            continue
                        new_cost = self.agent_costs.get(
                            (agent_uuid, o_type), 1.0
                        ) + running_cost
                        _, o_c = paths.get(o_type, ([], float('inf')))
                        if new_cost < o_c:
                            new_path = path.copy()
                            new_path.append(agent_uuid)
                            # the path length changed so no process in this
                            # loop. update path to shorter ver. and process
                            # next loop
                            if o_type in srcs:
                                srcs.remove(o_type)
                            paths[o_type] = (new_path, new_cost)
        wanted_paths = {}
        for dst in dsts:
            path, cost = paths.get(dst, (None, None))
            if path:
                wanted_paths[dst] = (tuple(path), cost)
        return wanted_paths

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
        for id_resolved in self.resolve(id_l):
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

    def resolve_document(self, documents: Iterable[dict],
                         in_map: Union[Dict[str, str],
                                       Callable[[dict], Dict[str, Any]]],
                         out_map: Union[str, Dict[str, str],
                                        Callable[[str, Any], Dict]]
                         ) -> Generator:
        """Resolve identifiers given a document

        Args:
            documents: a series of documents
            in_map:
            out_map:

        Returns:
            List of documents, with duplications when one set of
            identifiers gives more than one output of the desired
            identifier type.
        """
        # build input processor
        if callable(in_map):
            process_input = in_map
        else:
            def process_input(doc: dict):
                in_dict = {}
                for v_type, field in in_map.items():
                    v = dict_get_nested(document, field)
                    if v is None:
                        v = document.get(field, None)
                    if v is None:
                        continue
                    in_dict[v_type] = v
                return in_dict

        id_dicts = []
        o_docs = []
        for document in documents:
            o_docs.append(copy.deepcopy(document))
            id_dict = process_input(document)
            # end of extracting for all fields in single document
            id_dicts.append(id_dict)
        # end of extracting IDs from documents
        for od, res in zip(o_docs, self.resolve(id_dicts)):
            o_idv = od.get(self.document_resolve_id_field, None)
            for id_t in self.preferred:
                # FIXME: output according to out_map
                # FIXME: multiple output of same value type
                if id_t in res:
                    if len(res[id_t]) > 1:
                        self.logger.warning("%s->%s(multi)", o_idv, res[id_t])
                    for id_v in res[id_t]:
                        new_doc = copy.deepcopy(od)
                        new_doc[self.document_resolve_id_field] = id_v
                        self.logger.debug("updated doc ID %s->%s", o_idv, id_v)
                        yield new_doc
                    break
            else:  # did not break from loop == no new id
                self.logger.debug("did not update %s", o_idv)
                yield od  # no need to copy


class Decorators:
    """Decorators container

    """
    def __init__(self, parent):
        super(Decorators, self).__init__()
        self.parent = parent

    def resolve(self, func):
        """Decorator wrapper for :py:meth:`Resolver.resolve`,
        """
        func_name = 'resolve'
        if not hasattr(self.parent, func_name):
            raise AttributeError(f"Parent does not have {func_name}")

        @wraps(func)
        def wrapped(*args, **kwargs):
            input_docs = func(*args, **kwargs)
            yield from self.parent.resolve(input_docs)

        return wrapped

    def resolve_curie(self, func):
        """Decorator wrapper for :py:meth:`Resolver.resolve_curie`,
        """
        func_name = 'resolve_curie'
        if not hasattr(self.parent, func_name):
            raise AttributeError(f"Parent does not have {func_name}")

        @wraps(func)
        def wrapped(*args, **kwargs):
            input_docs = func(*args, **kwargs)
            yield from self.parent.resolve_curie(input_docs)

        return wrapped

    def resolve_document(self, func, *a, **kwa):
        """Decorator wrapper for :py:meth:`Resolver.resolve_document`,
        """
        func_name = 'resolve_document'
        if not hasattr(self.parent, func_name):
            raise AttributeError(f"Parent does not have {func_name}")

        @wraps(func)
        def wrapped(*args, **kwargs):
            input_docs = func(*args, **kwargs)
            yield from self.parent.resolve_document(input_docs, *a, **kwa)

        return wrapped

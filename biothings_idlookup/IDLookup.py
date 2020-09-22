# -*- coding: utf-8 -*-
""" IDLookup for obtaining different types of identifiers using existing identifiers

(some background info... TO BE WRITTEN)

One part of the core mechanism is implemented in the `IDLookup` class,
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
from collections import defaultdict
from functools import wraps
from typing import List, Dict, Iterable, Optional, Tuple, Sequence, Mapping, \
    Union, Callable, FrozenSet

from .containers import AgentsContainer, IDPropertyContainer, IDStructure
from .utils import dict_get_nested, transform_str


class IDLookup:
    """Base class for looking up IDs

    Attributes:
        preferred: List of output id_types in the order of preference
        batch_size: Size of documents to process in each batch

    """
    def __init__(self, input_types=None):
        super(IDLookup, self).__init__()
        self.logger = logging.getLogger('biothings_idlookup')
        self.preferred: List[str] = []
        self.in_fields: Dict[str, Union[str, Callable[[dict], str]]] = {}
        self.agents = AgentsContainer(cache_size=128)
        self.batch_size = 1000
        # now we store failed processors on a per id basis
        self.id_failed_edges = defaultdict(IDPropertyContainer)

        self.id_key_normalizer: Dict[str, Callable[[str], str]] = {}
        self.id_value_input_transforms: Dict[str, List[Callable[[str], Optional[str]]]] = {}
        self.id_value_output_transforms: Dict[str, List[Callable[[str], Optional[str]]]] = {}

        self.tracing = False
        self.last_batch = None

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

    def reset_id_failed_edges(self):
        self.id_failed_edges.clear()

    def add_id_key_normalizer(self, id_type: str, normalizer: Callable[[str], str]) -> None:
        """Set a id_value normalizer for id_type

        The result of this is used as keys in a dictionary to lookup properties related to id_type:id_value.
        Think of this as a easy way to get out of properly implementing a hash function.


        Examples:
            For instance, if id_values for id_type is case insensitive, then it is okay to do this
            >>> lookup = IDLookup()
            >>> lookup.add_id_key_normalizer('id_type', str.lower)

            Then 'ID_Value', 'id_Value', 'id_value' will all map to 'id_value' and share the same properties.

        Args:
            id_type: Type of identity
            normalizer: Function to normalize the id_value to keys.
        """
        self.reset_id_failed_edges()
        self.id_key_normalizer[id_type] = normalizer

    def add_id_transforms(self, direction: str, id_type: str, normalizer: Callable[[str], Optional[str]]) -> None:
        """Add id_value transformations on input/output, when transforming documents

        Args:
            direction: 'input' or 'output'
            id_type: type of identity to apply transform
            normalizer: transformation, return None if an id_value is invalid and needs to be removed

        Examples:
            For instance, if a document has may or may not have InChI ids, and pandas reading the document returns NaN,
            it is preferred to have them removed because they are invalid input and slows down lookup.
            >>> lookup = IDLookup()
            >>> lookup.add_id_transforms('input', 'inchi', lambda s: None if s != s else s)
            An input can also be skipped based on regex matching, see biothings_idlookup.transforms.filter_regex

            For instance, if the output of an PubChem identity needs to be prefixed with 'CID'
            >>> lookup = IDLookup()
            >>> lookup.add_id_transforms('output', 'pubchem', lambda s: f"CID {s}")

            Some common uses are implemented in biothings_idlookup.transforms and can be used as examples
            on implementing new transforms
        """
        # TODO: implement transforms that applies to ALL identity types (old behavior)
        if direction == 'input':
            self.id_value_input_transforms.setdefault(id_type, []).append(normalizer)
        elif direction == 'output':
            self.id_value_output_transforms.setdefault(id_type, []).append(normalizer)
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def get_shortest_path(self, failed_agents: frozenset,
                          available_sources: Iterable[str]) -> \
            Optional[List[str]]:
        """Find the path with lowest cost to most preferred destination

        Args:
            failed_agents:
            available_sources:

        Returns:

        """
        # build graph
        available_sources = set(available_sources)
        # return on first available destination in order of preference
        for destination in self.preferred:
            if destination in available_sources:
                # we already have what we want, skipping
                self.logger.debug("%s is already present, no path needed, returning None", destination)
                return None
            return self.agents.shortest_path(
                available_sources, destination, failed_agents)

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
                new_ids = self.lookup_documents(chunk_docs)
                for idx, doc in enumerate(chunk_docs):
                    out_doc = copy.deepcopy(doc)
                    new_id = new_ids[idx]
                    self.logger.debug("got new id %s", new_id)
                    if new_id is not None:
                        out_doc['_id'] = new_id
                    else:
                        pass
                    yield out_doc
            return None
        return wrapped_f

    def failed_agents(self, id_dict: Mapping[str, str]) -> FrozenSet[str]:
        failed = set()
        for id_type, id_value in id_dict.items():
            failed.update(
                self.id_failed_edges[id_type].get(id_value, set()))
        return frozenset(failed)

    def lookup_identifiers(self, ids: Sequence[Dict[str, str]]) \
            -> List[Optional[Tuple[str, Union[str, List[str]]]]]:
        """ Lookup preferred IDs given IDs, without performing transforms

        Args:
            ids: Dictionaries containing id_type:id_value
        Returns:
            List of IDs resulting from lookup, in the same order as input.
            For each element in the list, None means no result was found.
            A Tuple contains (id_type, id_values).
            id_values is either a str or a list of str, depending on settings
            for duplication/multiple results.
        """
        lookup_dicts = [IDStructure(id_dict) for id_dict in ids]
        for idstruct in lookup_dicts:
            idstruct.tracing = self.tracing
        output = [None] * len(ids)
        orig_indices: List[int] = list(range(len(lookup_dicts)))
        lookup_done = [False] * len(lookup_dicts)
        while not all(lookup_done):
            # Create a DAG for topological sort
            # here the 'edge' is a node in the graph
            agent_lookup_indices = {}
            dep_graph = {}
            for idx, id_dict in enumerate(lookup_dicts):
                if lookup_done[idx]:
                    self.logger.debug("%s is done, skip.", id_dict)
                    continue
                failed_agents = self.failed_agents(id_dict)
                avail_sources = id_dict.keys()
                path = self.get_shortest_path(failed_agents, avail_sources)
                if path is None:
                    # no path found (either no available path or we already have the result), mark as done
                    lookup_done[idx] = True
                    self.logger.debug("no path found")
                    continue
                path_length = len(path)
                self.logger.debug("path: %s", ",".join(path))
                path = path[::-1]  # reversed
                for agent_idx, agent_name in enumerate(path):
                    prerequisites = dep_graph.setdefault(agent_name, set())
                    if agent_idx + 1 < path_length:
                        prerequisites.add(path[agent_idx + 1])
                    agent_lookup_indices.setdefault(agent_name, []).append(idx)
            # sort DAG
            agents_order = []
            zero_indegree = [agent_name for agent_name, deps in
                             dep_graph.items() if len(deps) == 0]
            while zero_indegree:
                agent_name = zero_indegree.pop()
                agents_order.append(agent_name)
                del dep_graph[agent_name]
                for other_agent, deps in dep_graph.items():
                    if agent_name in deps:
                        deps.remove(agent_name)
                        # list is slightly faster in our use case so no sets
                        # and we don't want this to run multiple times..
                        if len(deps) == 0:
                            zero_indegree.append(other_agent)
            if dep_graph:
                raise RuntimeError("Loop in dependency graph for agents")
            for agent_name in agents_order:
                source, target, _, edge = self.agents.raw_agents[agent_name]
                # source_id_dict = {lookup_dicts[idx][source]: idx for idx in indices}
                source_id_dict = {}
                for idx in set(agent_lookup_indices[agent_name]):
                    source_id = lookup_dicts[idx].get(source, None)
                    if source_id is not None:
                        source_id_dict.setdefault(source_id, []).append(idx)
                results = edge.lookup(source_id_dict.keys())
                for source_id, indices in source_id_dict.items():
                    result = results.get(source_id, None)
                    if result is None:
                        self.id_failed_edges[source].setdefault(
                            source_id, set()
                        ).add(agent_name)
                        self.logger.debug("%s did not process %s:%s", agent_name, source, source_id)
                        # dequeue is probably more expensive
                        continue
                    if type(result) is list:
                        # TODO: implement duplication
                        raise NotImplementedError
                    result = str(result)  # force cast to string
                    for idx in indices:
                        lookup_dicts[idx].set_id_value(target, result, agent_name)
                        self.logger.debug("got %s:%s -> %s:%s", source, source_id, target, result)
            # end of while loop
        for lookup_idx, orig_idx in enumerate(orig_indices):
            for id_type in self.preferred:
                if id_type in lookup_dicts[lookup_idx]:
                    # TODO: implement duplication
                    output[orig_idx] = (id_type, lookup_dicts[lookup_idx][id_type])
                    break  # exit loop on first preferred id_type
        if self.tracing:
            self.last_batch = lookup_dicts
        return output

    def lookup_documents(self, documents: Sequence[dict]) -> List[Optional[str]]:
        """Lookup identities from a document, reading from document and applying transformations on input/output

        Args:
            documents:

        Returns:
            List of identifiers in the same order as the documents
        """
        id_dicts = []
        for document in documents:
            id_dict = {}  # {id_type: id_values}
            for id_type, field in self.in_fields.items():
                if isinstance(field, str):  # obtain value from single field
                    id_value = dict_get_nested(document, field)
                elif isinstance(field, Callable):  # obtain value if custom implementation is given
                    id_value = field(document)
                else:
                    raise RuntimeError("")  # TODO: add message
                if isinstance(id_value, str):
                    pass  # do nothing when we have str
                elif id_value is None:
                    continue  # skip when None is obtained
                else:
                    pass
                transforms = self.id_value_input_transforms.get(id_type, [])
                id_value = transform_str(id_value, transforms)
                if id_value is not None:
                    if not isinstance(id_value, str):
                        # self.logger.warning("got raw id_value:%r and force converting to str...", id_value)
                        id_value = str(id_value)
                        # self.logger.warning("...converted to %s", id_value)
                    id_dict[id_type] = id_value
            # end of extracting id_type:id_value for all fields in single document
            id_dicts.append(id_dict)
        # end of extracting IDs from documents
        output = []
        for result in self.lookup_identifiers(id_dicts):
            if isinstance(result, tuple):
                id_type, id_value = result
                if isinstance(id_value, list):
                    # TODO: handle duplication
                    output.append(id_value[0])
                else:
                    transforms = self.id_value_output_transforms.get(id_type, [])
                    id_value = transform_str(id_value, transforms)
                    output.append(id_value)
            else:
                output.append(None)  # got no preferred result
        return output

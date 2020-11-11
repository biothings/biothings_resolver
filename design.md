# Resolver Design Documentation

Last Updated on November 11, 2020.



## `Resolver` Class

### Attributes

#### General Attributes

Controls general behavior during input resolution.

##### `debug`

Type:`bool`. Controls whether debugging information will be output using the standard logging utilities. It will be the same as setting up logging manually, this is only a convenience feature.

To observe how a single input is resolved, set `debug` to `True` and invoke any resolving functions with a single input.

##### `preferred`

Type: `List[str]`. A list of preferred output types, in the order of preference, CURIE-prefix-style-like.

##### `expand`

Type:`bool`. Whether any resolving functions will attempt to return all the preferred output types, or only return the available most preferred output type.

##### `batch_size`

Type:`int`.  The maximum number of items that a resolving agent will be asked to handle at once. An agent may be asked to process less number of items than `batch_size` at one time, but will never be asked to process more items.

#### Transform Attributes

Values may go through additional transformations before and/or after being processed by agents. The following attributes control this transform

##### `value_in_xfrm`, `value_out_xfrm`, `curie_in_xfrm`, `curie_out_xfrm`

Type: `Dict[str, Callable]`.

The keys can be CURIE-prefix like, or a single asterisk('\*'). A CURIE-prefix like key will only apply the transformations to the specific types of values, an asterisk will apply to values of all types.

The values are functions that takes a single input and produces a single output, used to transform the values, if `None` is returned, then the value is discarded.

 `value_*` transforms apply to all resolving functions except for CURIE related ones, `curie_*` transforms apply to only CURIE related resolving functions.

 `curie_in_xfrm` will receive the original CURIE-like input, for instance when the input CURIE is "CHEBI:27732", the corresponding transform function will receive "CHEBI:27732" as its input.

`*_in_*` transforms are applied before a value is processed by any agents, `*_out_*` transforms are applied after all processing, right before output.

To perform multiple transforms, compose a function that performs multiple transforms.

Example use cases: 1) prepend/remove "CID" prefixes for PubChem CID values, 2) prepend/remove "CHEBI:" when using ChEBI IDs in CURIE resolving.

### Methods

#### `resolve(Sequence[Dict[str, Any]]) -> Generator[Dict[str, list], None, None]`

Takes in an ordered collection (for instance, a `list`) of dictionaries. Each dictionary represents a group of inputs, combined together in a similar fashion to a logical OR operation, which means an input of any type in the group will be used for resolving to the desired output types. The keys of the input dictionary is a CURIE-prefix for the corresponding type.

The output is in the same order as the input. The dictionary will have keys specified in `Resolver.preferred`, instead of the canonical version. For instance, if "fb" is an alias to the canonical prefix "FLYBASE", and "fb" is specified in the `Resolver.preferred` attribute, then the output dictionary will use "fb" as its key when an output value is of type "FLYBASE". If any results are produced for a type of output value, it will be stored in the `list`, which has at least 1 element (i.e., no key if no result, singular results will not be taken out of  `list` container). If a group of input values does not produce an output, the output will be an empty dictionary.

Only the resolution results will be in the output (as configured in `Resolver.preferred` and `Resolver.expand`). To copy other fields from the input, use the `resolve_document` method below.

#### `resolve_curie(Sequence[str]) -> Generator[List[str], None, None]`

Takes in an ordered collection of CURIEs. Each CURIE is an input.

Produces `list`s of `str`, in the same order as the input. The CURIE prefix in the output will use the version specified in `Resolver.preferred` instead of the canonical version. If an input has multiple output values of the same type, they will all be present in the output `list`. If an input does not produce an output, its corresponding output will be an empty `list`.

#### `resolve_document(Sequence[dict], in_map=Dict[str, Any], out_map=Union[str, Dict[str, Any]])`

Takes in an order collection of dictionaries, extracting input values and types from its specific fields according to `in_map`, performs resolution on the input, then copies and updates the dictionary with the output values according to `out_map`.

##### `in_map`

A dictionary. Keys are CURIE-prefixes of the corresponding type. Values point to the corresponding field in the dictionary. If the value is a `Callable`, the entire dictionary will be passed to the function, and its output will be used as the value for the corresponding type. If the value is a `str` and contains dots ('.') then resolver will attempt to read a nested dictionary, if that fails, it will attempt to read using the entire key, i.e., resolver will try to read `in_dict['nested']['value']['here']` and when that fails it will try `in_dict['nested.value.here']`. To address anything not covered here, use a `Callable`.

##### `out_map`

A `str` or a `dict` that maps CURIE-prefixes of the corresponding type to the corresponding field in the dictionary to write to. Dot notations will not be expanded to nested dictionaries, to do that, consider using transforms creatively.

If the `expand` attribute is set to `False`, then `out_map` may be a `str` that points to the field that any output type will write to.



### Decorators

Decorators methods are accessible in `Resolver.decorators`.

#### `resolve`



#### `resolve_curie`



#### `resolve_document`
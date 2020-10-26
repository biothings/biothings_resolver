import re
from typing import Tuple


# We support a CURIE-like syntax
# CURIE is Compact URI and its syntax is documented here:
# https://www.w3.org/TR/curie/#ref_IRI .
# Basically it's a prefix and a reference separated by a colon (':').
# The prefix is an XML name without colons, and the reference is part
# of an IRI or basically internationalized URI.
# Supporting the real thing would be way too complicated, and furthermore
# we want case-insensitive (but preserving) matching which is actually
# very complex when taking locales into account, hence here we
# add further restrictions to the prefix, but relax the requirements
# for the reference: the prefix has to be 7-bit ASCII and cannot be omitted,
# but the reference can be any non-empty string.
#
# Another note on ASCII: Supposedly if we want to support full ASCII:
# regex_ncname_like = re.compile(r"""
# \s*                      # ignore leading whitespace
# [A-Za-zÀ-ÖØ-öø-ÿ_]       # Letters | '_'
# [A-Za-zÀ-ÖØ-öø-ÿ0-9.-_·]* # Letters | Digit | '.' | '-' | '_' | Extender
# """, re.ASCII | re.VERBOSE)
# but this doesn't actually work as intended, it matches other characters
# for whatever reasons, makes things harder to debug, and is probably
# not very useful in our use cases.

regex_prefix = re.compile(r'[A-Za-z_][\w.-]*', re.ASCII)
regex_prefix_ref = re.compile(r'(\[)?([A-Za-z_][\w.-]*?):(.+)(?(1)]|$)',
                              re.UNICODE)


def split_curie(curie: str) -> Tuple[str, str]:
    m = regex_prefix_ref.match(curie)
    if m is None:
        raise ValueError("Malformed CURIE-like input")
    _, prefix, reference = m.groups()
    # check for ASCII: Python re does not support inline flags properly
    if not validate_prefix(prefix):
        raise ValueError(f"Non-ASCII prefix ({prefix}) is unsupported")
    return prefix, reference


def validate_prefix(prefix: str) -> bool:
    if regex_prefix.fullmatch(prefix):
        return True
    else:
        return False

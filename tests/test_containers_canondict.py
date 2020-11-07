import pytest

from biothings_resolver.containers import CanonDict


def test_len():
    od = {'A': 1}
    d = CanonDict(od)
    d.add_alias('first_letter', 'A')
    assert len(d) == len(od)


def test_iter():
    od = {'A': 'a', 'B': 2}
    d = CanonDict(od)
    dk = set()
    for k in d:
        dk.add(k)
    assert set(od) == set(dk)


def test_toggle_case_sensitive():
    d = CanonDict()
    d.case_sensitive = True
    assert d.case_sensitive is True
    d.case_sensitive = False
    assert d.case_sensitive is False


def test_toggle_case_sensitive_raises():
    d = CanonDict({'A': 'A', 'a': 'a'})
    with pytest.raises(RuntimeError):
        d.case_sensitive = False


def test_non_letter_set_get():
    d = CanonDict()
    d[1] = 1
    assert d[1] == 1
    d.case_sensitive = False
    d[2] = '2'
    assert d[2] == '2'


def test_add_alias():
    d = CanonDict(A='1111')
    d.add_alias('first_letter', 'A')
    assert d['first_letter'] == d['A']


def test_del_alias():
    d = CanonDict(A='1111')
    d.add_alias('first_letter', 'A')
    d.delete_alias('first_letter')
    with pytest.raises(KeyError):
        _ = d['first_letter']


def test_add_alias_raises_on_already_exist():
    d = CanonDict(A='1', B='2')
    with pytest.raises(KeyError):
        d.add_alias('B', 'A')


def test_add_alias_raises_on_no_dst():
    d = CanonDict(A='1')
    with pytest.raises(KeyError):
        d.add_alias('second_letter', 'B')


def test_canon_key_new_key():
    d = CanonDict()
    k = 'k'
    assert d.get_canon_key(k) == k


def test_canon_key_casing():
    d = CanonDict(A=1)
    d.case_sensitive = False
    assert d.get_canon_key('a') == 'A'


def test_del_item():
    d = CanonDict(A=1)
    del d['A']
    with pytest.raises(KeyError):
        _ = d['A']


def test_del_via_alias():
    d = CanonDict(A=1)
    d.add_alias('first_letter', 'A')
    del d['first_letter']
    with pytest.raises(KeyError):
        _ = d['A']
    with pytest.raises(KeyError):
        _ = d['first_letter']


# not doing the Turkey test regarding case-insensitivity

# -*- coding: utf-8 -*-
import sys
import re

import vaex
import numpy as np
import pyarrow as pa
import pytest


@pytest.mark.skipif(vaex.utils.osname == 'windows',
                    reason="windows' snprintf seems buggy")
def test_format():
    num1 = np.array([1, 2, 3], dtype=np.int32)
    num2 = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    text = ['Here', 'we', 'go']

    df = vaex.from_arrays(num1=num1, num2=num2, text=text)

    assert df.num1.format("%d").tolist() == ['1', '2', '3']
    assert df.num1.format("%04d").tolist() == ['0001', '0002', '0003']
    assert df.num2.format('%f').tolist() == ['1.100000', '2.200000', '3.300000']
    assert df.num2.format('%05.2f').tolist() == ['01.10', '02.20', '03.30']
    assert df.text.format('pre-%s-post').tolist() == ['pre-%s-post' % k for k in text]


@pytest.mark.skipif(sys.version_info < (3, 3), reason="requires python3.4 or higher")
def test_dtype_object_string(tmpdir):
    x = np.arange(8, 12)
    s = np.array(list(map(str, x)), dtype='O')
    df = vaex.from_arrays(x=x, s=s)
    assert df.columns['s'].dtype.kind == 'O'
    path = str(tmpdir.join('test.arrow'))
    df.export(path)
    df_read = vaex.open(path, as_numpy=False)
    # the data type of s can be different
    assert df_read.compare(df) == ([], [], [], [])


def test_export_arrow_strings_to_hdf5(tmpdir):
    df = vaex.from_arrays(names=np.array(['hi', 'is', 'l2', np.nan], dtype='O'))
    path = str(tmpdir.join('test.arrow'))
    df.export(path)
    df_read_arrow = vaex.open(path, as_numpy=False)
    path = str(tmpdir.join('test.hdf5'))
    df.export(path)
    df_read_hdf5 = vaex.open(path)
    assert df_read_hdf5.compare(df_read_arrow) == ([], [], [], [])


def test_arrow_strings_concat(tmpdir):
    df = vaex.from_arrays(names=['hi', 'is', 'l2'])
    path = str(tmpdir.join('test.arrow'))
    df.export(path)
    df_read_arrow = vaex.open(path, as_numpy=False)
    path = str(tmpdir.join('test.hdf5'))
    df_read_arrow.export(path)
    df_read_hdf5 = vaex.open(path)
    assert df_read_hdf5.compare(df_read_arrow) == ([], [], [], [])


def test_concat():
    ds1 = vaex.from_arrays(names=['hi', 'is', 'l2'])
    ds2 = vaex.from_arrays(names=['hello', 'this', 'is', 'long'])
    ds = ds1.concat(ds2)
    assert len(ds) == len(ds1) + len(ds2)
    assert ds.data_type('names') == pa.string()
    assert ds.data_type('names') != np.object


def test_string_count_stat():
    ds = vaex.from_arrays(names=['hello', 'this', 'is', 'long'])
    assert ds.count(ds.names) == 4
    ds = vaex.from_arrays(names=np.ma.array(['hello', 'this', 'is', 'long'], mask=[0, 0, 1, 0]))
    assert ds.count(ds.names) == 3
    df = vaex.from_arrays(names=np.array(['hi', 'is', 'l2', np.nan], dtype='O'))
    assert df.count(ds.names) == 3

    names = vaex.string_column(['hello', 'this', None, 'long'])
    x = np.arange(len(names))
    df = vaex.from_arrays(names=names, x=x)
    assert df.count(df.names, binby='x', limits=[0, 100], shape=1).tolist() == [3]


@pytest.mark.skip
def test_string_dtype_with_none():
    ds = vaex.from_arrays(names=['hello', 'this', 'is', None])
    assert ds.count(ds.names) == 3


def test_unicode():
    ds = vaex.from_arrays(names=['bla\u1234'])
    assert ds.names.dtype == pa.string()
    ds = vaex.from_arrays(names=['bla'])
    assert ds.names.dtype == pa.string()


@pytest.mark.skipif(sys.version_info < (3, 3), reason="requires python3.4 or higher")
def test_concat_mixed():
    # this can happen when you want to concat multiple csv files
    # and pandas makes one have nans, since they all have missing values
    # and the other string
    ds1 = vaex.from_arrays(names=['not', 'missing'])
    ds2 = vaex.from_arrays(names=[np.nan, np.nan])
    assert ds1.data_type(ds1.names) == pa.string()
    assert ds2.data_type(ds2.names) == np.float64
    ds = ds1.concat(ds2)
    assert len(ds) == len(ds1) + len(ds2)
    assert ds.data_type(ds.names) == ds1.names.dtype

def test_strip():
    ds = vaex.from_arrays(names=['this ', ' has', ' space'])
    ds['stripped'] = ds.names.str.strip()
    ds.stripped.tolist() == ['this', 'has', 'space']


@pytest.mark.skipif(sys.version_info < (3, 3), reason="requires python3.4 or higher")
def test_unicode2(tmpdir):
    path = str(tmpdir.join('utf32.hdf5'))
    ds = vaex.from_arrays(names=["vaex", "or", "væx!"])
    assert ds.names.dtype == pa.string()
    ds.export_hdf5(path)
    ds = vaex.open(path)
    assert ds.names.dtype == pa.string()
    assert ds.names.tolist() == ["vaex", "or", "væx!"]


@pytest.fixture(params=['dfs_arrow', 'dfs_array'])
def dfs(request, dfs_arrow, dfs_array):
    named = dict(dfs_arrow=dfs_arrow, dfs_array=dfs_array)
    return named[request.param]


string_list = ["vaex", " \tor", "VæX! ", "vaex or VÆX!", "Æ and", "æ are weird", "12", "æ", "a1", "a1æ", "\t "]
unicode_compat = lambda x: x
try:
    unicode
    unicode_compat = lambda x: x.decode('utf8')
    string_list = map(unicode_compat, string_list)
except NameError:
    pass
string_list_reverse = string_list[::-1]


@pytest.fixture(scope='session')
def dfs_arrow(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("vaex")
    path = str(tmpdir / 'strings.hdf5')
    df = vaex.from_arrays(s=vaex.string_column(string_list), sr=vaex.string_column(string_list_reverse))
    df.export(path)  # we write it out so that the memory is read only
    return vaex.open(path)


def test_null_values():
    df = vaex.from_arrays(s=vaex.string_column(['aap', None, 'mies']), x=[0, 1, 2])
    assert df.count() == 3
    assert df.count(df.s) == 2
    assert df.count(df.s, selection=df.x > 0) == 1


@pytest.fixture()
def dfs_array():
    return vaex.from_arrays(s=np.array(string_list, dtype='O'), sr=np.array(string_list_reverse, dtype='O'))


def test_byte_length(dfs):
    assert dfs.s.str.byte_length().tolist() == [len(k.encode('utf8')) for k in string_list]


def test_string_capitalize(dfs):
    assert dfs.s.str.capitalize().tolist() == dfs.s.str_pandas.capitalize().tolist()


def test_string_cat(dfs):
    c = [s1+s2 for s1, s2 in zip(string_list, string_list_reverse)]
    assert dfs.s.str.cat(dfs.sr).tolist() == c
    assert dfs.s.str_pandas.cat(dfs.sr).tolist() == c


def test_string_contains(dfs):
    assert dfs.s.str.contains('v', regex=False).tolist() == [True, False, False, True, False, False, False, False, False, False, False]
    assert dfs.s.str.contains('æ', regex=False).tolist() == [False, False, True, False, False, True, False, True, False, True, False]
    assert dfs.s.str.contains('Æ', regex=False).tolist() == [False, False, False, True, True, False, False, False, False, False, False]


@pytest.mark.parametrize("width", [2, 10])
def test_string_center(dfs, width):
    assert dfs.s.str.center(width).tolist() == dfs.s.str_pandas.center(width).tolist()


def test_string_counts(dfs):
    assert dfs.s.str.count("v", regex=False).tolist() == dfs.s.str_pandas.count("v").tolist()
    assert dfs.s.str.count("[va]", regex=True).tolist() == dfs.s.str_pandas.count("[va]").tolist()


def test_string_endswith(dfs):
    assert dfs.s.str.endswith("x").tolist() == dfs.s.str_pandas.endswith("x").tolist()


@pytest.mark.parametrize("sub", ["v", unicode_compat("æ")])
@pytest.mark.parametrize("start", [0, 3, 5])
@pytest.mark.parametrize("end", [-1, 3, 5, 10])
def test_string_find(dfs, sub, start, end):
    assert dfs.s.str.find(sub, start, end).tolist() == dfs.s.str_pandas.find(sub, start, end).tolist()


@pytest.mark.parametrize("i", [-1, 3, 5, 10])
def test_string_get(dfs, i):
    x = dfs.s.str_pandas.get(i).values.tolist()
    assert dfs.s.str.get(i).tolist() == [k[i] if i < len(k) else '' for k in string_list]


@pytest.mark.parametrize("sub", ["v", "æ"])
@pytest.mark.parametrize("start", [0, 3, 5])
@pytest.mark.parametrize("end", [-1, 3, 5, 10])
def test_string_index(dfs, sub, start, end):
    assert dfs.s.str.find(sub, start, end).tolist() == dfs.s.str.index(sub, start, end).tolist()


@pytest.mark.parametrize("pattern", [None, ' '])
def test_string_join(dfs, pattern):
    assert dfs.s.str.split(pattern).str.join('-').tolist() == dfs.s.str.split(pattern).str.join('-').tolist()


def test_string_len(dfs):
    assert dfs.s.str.len().astype('i4').tolist() == [len(k) for k in string_list]
    assert dfs.s.str_pandas.len().astype('i4').tolist() == [len(k) for k in string_list]


@pytest.mark.parametrize("width", [2, 10])
def test_string_ljust(dfs, width):
    assert dfs.s.str.ljust(width).tolist() == dfs.s.str_pandas.ljust(width).tolist()


def test_string_lower(dfs):
    assert dfs.s.str.lower().tolist() == dfs.s.str_pandas.lower().tolist()


def test_string_lstrip(dfs):
    assert dfs.s.str.lstrip().tolist() == dfs.s.str_pandas.lstrip().tolist()
    assert dfs.s.str.lstrip('vV ').tolist() == dfs.s.str_pandas.lstrip('vV ').tolist()


def test_string_match(dfs):
    assert dfs.s.str.match('^v.*').tolist() == dfs.s.str_pandas.match('^v.*').tolist()
    assert dfs.s.str.match('^v.*').tolist() == [k.startswith('v') for k in string_list]

# TODO: normalize


@pytest.mark.parametrize("width", [2, 10])
@pytest.mark.parametrize("side", ['left', 'right', 'both'])
def test_string_pad(dfs, width, side):
    assert dfs.s.str.pad(width, side=side).tolist() == dfs.s.str_pandas.pad(width, side=side).tolist()

# TODO: partition


@pytest.mark.parametrize("repeats", [1, 3])
def test_string_repeat(dfs, repeats):
    assert dfs.s.str.repeat(repeats).tolist() == dfs.s.str_pandas.repeat(repeats).tolist()


@pytest.mark.parametrize("pattern", ["v", " ", unicode_compat("VæX")])
@pytest.mark.parametrize("replacement", ["?", unicode_compat("VæX")])
@pytest.mark.parametrize("n", [-1, 1])
def test_string_replace(dfs, pattern, replacement, n):
    assert dfs.s.str.replace(pattern, replacement, n).tolist() == dfs.s.str_pandas.replace(pattern, replacement, n).tolist()


@pytest.mark.parametrize("pattern", ["v", " "])
@pytest.mark.parametrize("replacement", ["?", unicode_compat("VæX")])
@pytest.mark.parametrize("flags", [0, int(re.IGNORECASE)])
def test_string_replace_regex(dfs, pattern, replacement, flags):
    assert dfs.s.str.replace(pattern, replacement, flags=flags, regex=True).tolist() == \
           dfs.s.str_pandas.replace(pattern, replacement, flags=flags, regex=True).tolist()


@pytest.mark.xfail(reason='unicode not supported fully in regex')
@pytest.mark.parametrize("pattern", [unicode_compat("VæX")])
@pytest.mark.parametrize("replacement", ["?", unicode_compat("VæX")])
@pytest.mark.parametrize("flags", [0, int(re.IGNORECASE)])
def test_string_replace_regex_unicode(dfs, pattern, replacement, flags):
    assert dfs.s.str.replace(pattern, replacement, flags=flags, regex=True).tolist() == \
           dfs.s.str_pandas.replace(pattern, replacement, flags=flags, regex=True).tolist()


@pytest.mark.parametrize("sub", ["v", unicode_compat("æ")])
@pytest.mark.parametrize("start", [0, 3, 5])
@pytest.mark.parametrize("end", [-1, 3, 5, 10])
def test_string_rfind(dfs, sub, start, end):
    assert dfs.s.str.rfind(sub, start, end).tolist() == dfs.s.str_pandas.rfind(sub, start, end).tolist()


@pytest.mark.parametrize("sub", ["v", unicode_compat("æ")])
@pytest.mark.parametrize("start", [0, 3, 5])
@pytest.mark.parametrize("end", [-1, 3, 5, 10])
def test_string_rindex(dfs, sub, start, end):
    assert dfs.s.str.rindex(sub, start, end).tolist() == dfs.s.str_pandas.rfind(sub, start, end).tolist()


@pytest.mark.parametrize("width", [2, 10])
def test_string_rjust(dfs, width):
    assert dfs.s.str.rjust(width).tolist() == dfs.s.str_pandas.rjust(width).tolist()


def test_string_rstrip(dfs):
    assert dfs.s.str.rstrip().tolist() == dfs.s.str_pandas.rstrip().tolist()
    assert dfs.s.str.rstrip('x! ').tolist() == dfs.s.str_pandas.rstrip('x! ').tolist()


# @pytest.mark.parametrize("start", [0, 3, 5])
# @pytest.mark.parametrize("end", [-1, 3, 5, 10])
@pytest.mark.parametrize("start", [0, -1, -5, 10])
@pytest.mark.parametrize("end", [None, -1, 3, 1000])
def test_string_slice(dfs, start, end):
    assert dfs.s.str.slice(start, end).tolist() == dfs.s.str_pandas.slice(start, end).tolist()


def test_string_startswith(dfs):
    assert dfs.s.str.startswith("x").tolist() == dfs.s.str_pandas.startswith("x").tolist()


def test_string_strip(dfs):
    assert dfs.s.str.rstrip().tolist() == dfs.s.str_pandas.rstrip().tolist()
    assert dfs.s.str.rstrip('vx! ').tolist() == dfs.s.str_pandas.rstrip('vx! ').tolist()


def test_string_title(dfs):
    assert dfs.s.str.title().tolist() == dfs.s.str_pandas.title().tolist()


def test_string_lower(dfs):
    assert dfs.s.str.lower().tolist() == dfs.s.str_pandas.lower().tolist()


def test_string_upper(dfs):
    assert dfs.s.str.upper().tolist() == dfs.s.str_pandas.upper().tolist()


def test_string_isalnum(dfs):
    assert dfs.s.str.isalnum().tolist() == dfs.s.str_pandas.isalnum().tolist()


def test_string_isalpha(dfs):
    assert dfs.s.str.isalpha().tolist() == dfs.s.str_pandas.isalpha().tolist()


def test_string_isdigit(dfs):
    assert dfs.s.str.isdigit().tolist() == dfs.s.str_pandas.isdigit().tolist()


def test_string_isspace(dfs):
    assert dfs.s.str.isspace().tolist() == dfs.s.str_pandas.isspace().tolist()


def test_string_islower(dfs):
    assert dfs.s.str.islower().tolist() == dfs.s.str_pandas.islower().tolist()
    assert dfs.s.str.lower().str.islower().tolist() == dfs.s.str_pandas.lower().str_pandas.islower().tolist()


def test_string_isupper(dfs):
    assert dfs.s.str.isupper().tolist() == dfs.s.str_pandas.isupper().tolist()
    assert dfs.s.str.upper().str.isupper().tolist() == dfs.s.str_pandas.upper().str_pandas.isupper().tolist()


# def test_string_istitle(dfs):
#   assert dfs.s.str.istitle().tolist() == dfs.s.str_pandas.istitle().tolist()
#   assert dfs.s.str.title.istitle().tolist() == dfs.s.str_pandas.title().str_pandas.istitle().tolist()


def test_string_isspace(dfs):
    assert dfs.s.str.isspace().tolist() == dfs.s.str_pandas.isspace().tolist()


@pytest.mark.parametrize("width", [2, 10])
def test_string_zfill(dfs, width):
    assert dfs.s.str.zfill(width).tolist() == dfs.s.str_pandas.zfill(width).tolist()


def test_to_string():
    x = np.arange(1, 4, dtype='f4')
    df = vaex.from_arrays(x=x)
    df['s'] = df.x.to_string()
    assert df.s.tolist() == ["%f" % k for k in x]


def test_string_strip_special_case():
    strings = ["Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? "
               "They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. "
               "And please don't remove the template from the talk page since I'm retired now.89.205.38.27"]
    df = vaex.from_arrays(s=vaex.string_column(strings))
    df.s.str.strip(' ').values  # .get(0)


def test_string_strip_special_case2():
    strings = ['The eunuch in question left me no choice but to reinsert it. Take action as you see fit.·snunɐw·']
    df = vaex.from_arrays(s=vaex.string_column(strings))
    assert df.s.str.upper().tolist() == df.s.str_pandas.upper().tolist()


@pytest.mark.xfail(reason='we need to fix this, similar to upper and lower')
def test_string_strip_special_case3():
    strings = ['ɐa', 'aap']
    df = vaex.from_arrays(s=vaex.string_column(strings))
    assert df.s.str.capitalize().tolist() == df.s.str_pandas.capitalize().tolist()


def test_string_slice_repr():
    s = ['Here', 'is', 'a', 'simple', 'unit-test']
    df = vaex.from_arrays(s=s)
    df['sliced_s'] = df.s.str.slice(start=2, stop=5)
    repr(df['sliced_s'])


@pytest.mark.skipif(sys.version_info[0] == 2, reason="no support for python2")
@pytest.mark.parametrize("match", ["vaex", "VæX! "])
def test_strings_operator_equals(dfs, match):
    assert (dfs.s == match).tolist() == [k == match for k in string_list]
    assert (match == dfs.s).tolist() == [k == match for k in string_list]
    assert (dfs.s == dfs.s).tolist() == [k == k for k in string_list]


@pytest.mark.skipif(sys.version_info[0] == 2, reason="no support for python2")
@pytest.mark.parametrize("extra", ["vaex", "VæX! "])
def test_strings_operator_plus(dfs, extra):
    assert (dfs.s + extra).tolist() == [k + extra for k in string_list]
    assert (extra + dfs.s).tolist() == [extra + k for k in string_list]

    assert (dfs.s + dfs.s).tolist() == [k + k for k in string_list]
    assert (dfs.s + extra + dfs.s).tolist() == [k + extra + k for k in string_list]


def test_masked_string():
    s = np.ma.MaskedArray(data=['dog', 'dog', 'cat', 'cat', 'mouse'], mask=[False, False, True, False, True])
    df = vaex.from_arrays(s=s)
    assert (df.s == 'cat').tolist() == [False, False, False, True, False]


def test_string_operations_from_mmap_file(tmpdir):
    # if we write the file to disk and mmap it read only, we trigger invalid memory writes
    # see https://github.com/vaexio/vaex/pull/459
    x = np.arange(5)
    y = np.array(['This', 'is', 'a', None, 'test'])
    df = vaex.from_arrays(x=x, y=y)
    filename = str(tmpdir / 'test.hdf5')
    df.export_hdf5(filename)
    df_from_file = vaex.open(filename)
    assert df_from_file.y.str.slice(start=0, stop=2).tolist() == ['Th', 'is', 'a', None, 'te']
    assert df_from_file.y.str.upper().tolist() == ['THIS', 'IS', 'A', None, 'TEST']

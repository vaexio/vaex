from vaex.superutils import *
import vaex.strings
from vaex.utils import dropnan
import numpy as np
import sys
import pytest
import pyarrow as pa

@pytest.mark.parametrize('counter_cls', [counter_string]) #, counter_stringview])
def test_counter_string(counter_cls):
    strings = vaex.strings.array(['aap', 'noot', 'mies'])
    counter = counter_cls(1)
    counter.update(strings)
    counts = counter.extract()[0]

    assert counts["aap"] == 1
    assert counts["noot"] == 1
    assert counts["mies"] == 1

    strings2 = vaex.strings.array(['aap', 'n00t'])
    counter.update(strings2)
    counts = counter.extract()[0]
    assert counts["aap"] == 2
    assert counts["noot"] == 1
    assert counts["n00t"] == 1
    assert counts["mies"] == 1

    # test merge
    strings1 = vaex.strings.array(['aap', 'noot', 'mies'])
    counter1 = counter_cls(1)
    counter1.update(strings1)

    strings2 = vaex.strings.array(['kees', None])
    counter2 = counter_cls(1)
    counter2.update(strings2)
    counter1.merge(counter2)
    assert set(counter1.key_array().tolist()) == {'aap', 'noot', 'mies', 'kees', None}
    assert counter1.counts().tolist() == [1, 1, 1, 1, 1]


def test_counter_string_nulls_issue():
    first = [None, 'b', 'c']
    strings = vaex.strings.array(first)
    counter = counter_string(1)
    counter.update(strings)
    more = ['d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
    strings2 = vaex.strings.array(more)
    # the second update would nut grow the null array correctly
    counter.update(strings2)
    assert set(counter.key_array().tolist()) == set(first + more)


def test_set_string():
    set = ordered_set_string(1)
    strings = ['aap', 'noot', 'mies']
    strings_array = vaex.strings.array(strings)
    set.update(strings_array)
    set.seal()
    assert set.keys() == strings

    keys = set.key_array()
    # col = vaex.column.ColumnStringArrow.from_string_sequence(keys)
    # keys = pa.array(col)
    keys = pa.array(keys.to_numpy())
    assert keys.tolist() == strings


def test_set_bool():
    bset = ordered_set_bool(4)
    ar = np.array([True, True, False, False, True])
    chunk_size = 1024**2
    bset.update(ar, -1, chunk_size=chunk_size, bucket_size=chunk_size*4)
    keys = bset.key_array()
    assert len(keys) == 2
    assert set(keys.tolist()) == {True, False}


@pytest.mark.parametrize("nan", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("nmaps", [1, 2, 3])
def test_set_float(repickle, nan, missing, nmaps):
    ar = np.arange(4, dtype='f8')[::-1].copy()
    keys_expected = [3, 2, 1, 0]
    null_index = 2
    if missing:
        mask = [0, 0, 1, 0]
        keys_expected[null_index] = None
    if nan:
        ar[1] = np.nan
        keys_expected[1] = np.nan
    oset = ordered_set_float64(nmaps)
    if missing:
        ordinals_local, map_index = oset.update(ar, mask, return_values=True)
    else:
        ordinals_local, map_index = oset.update(ar, return_values=True)
    ordinals = np.empty(len(keys_expected), dtype='i8')
    ordinals = oset.flatten_values(ordinals_local, map_index, ordinals)
    keys = oset.keys()
    # if missing:
    #     ordinals[oset.null_index] = oset.null_index
    assert dropnan(np.take(keys, ordinals).tolist()) == dropnan(keys_expected)

    # plain object keys
    oset.seal()
    keys = oset.keys()
    expect_nan = 1 if nan else None
    assert dropnan(set(keys), expect=expect_nan) == dropnan(set(keys_expected), expect=expect_nan)
    assert oset.map_ordinal(keys).dtype.name == 'int8'

    # arrays
    keys = oset.key_array().tolist()
    if missing:
        keys[oset.null_index] = None
    assert dropnan(set(keys), expect=expect_nan) == dropnan(set(keys_expected), expect=expect_nan)
    if nan:
        assert np.isnan(keys[oset.nan_index])
    ordinals = oset.map_ordinal(keys).tolist()
    if missing:
        ordinals[oset.null_index] = oset.null_index
    assert ordinals == list(range(4))

    # tests extraction and constructor
    keys = oset.key_array()
    set_copy = ordered_set_float64(keys, oset.null_index, oset.nan_count, oset.null_count, '')
    keys = set_copy.key_array().tolist()
    if missing:
        keys[oset.null_index] = None
    assert dropnan(set(keys)) == dropnan(set(keys_expected))
    if nan:
         assert np.isnan(keys[set_copy.nan_index])
    ordinals = set_copy.map_ordinal(keys).tolist()
    if missing:
        ordinals[set_copy.null_index] = set_copy.null_index
    assert ordinals == list(range(4))

    # test pickle
    set_copy = repickle(oset)
    keys = set_copy.key_array().tolist()
    if missing:
        keys[oset.null_index] = None
    assert dropnan(set(keys)) == dropnan(set(keys_expected))
    if nan:
        assert np.isnan(keys[set_copy.nan_index])
    ordinals = set_copy.map_ordinal(keys).tolist()
    if missing:
        ordinals[set_copy.null_index] = set_copy.null_index
    assert ordinals == list(range(4))



@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("nmaps", [1, 2, 3])
def test_set_string(repickle, missing, nmaps):
    ar = ["aap", "noot", "mies", "teun"]
    keys_expected = ar
    null_index = 1
    if missing:
        ar[null_index] = None
        keys_expected[null_index] = None
        keys_expected = ar
    arlist = ar
    ar = vaex.strings.array(ar)
    oset = ordered_set_string(nmaps)
    ordinals_local, map_index = oset.update(ar, return_values=True)
    ordinals = np.empty(len(keys_expected), dtype='i8')
    ordinals = oset.flatten_values(ordinals_local, map_index, ordinals)
    keys = oset.key_array()
    assert keys.to_numpy()[ordinals].tolist() == keys_expected

    # plain object keys
    oset.seal()
    keys = oset.keys()
    assert set(keys) == set(keys_expected)
    # return
    assert oset.map_ordinal(vaex.strings.array(keys)).dtype.name == 'int8'
    # return

    # arrays
    keys = oset.key_array()
    assert set(keys.tolist()) == set(keys_expected)
    ordinals = oset.map_ordinal(keys).tolist()
    if missing:
        ordinals[oset.null_index] = oset.null_index
    assert ordinals == list(range(4))

    # tests extraction and constructor
    keys = oset.key_array()
    set_copy = ordered_set_string(keys, oset.null_index, oset.nan_count, oset.null_count, '')
    keys = set_copy.key_array()
    assert set(keys.tolist()) == set(keys_expected)
    ordinals = set_copy.map_ordinal(keys).tolist()
    if missing:
        ordinals[set_copy.null_index] = set_copy.null_index
    assert ordinals == list(range(4))

    # test pickle
    set_copy = repickle(oset)
    keys = set_copy.key_array()
    assert set(keys.tolist()) == set(keys_expected)
    ordinals = set_copy.map_ordinal(keys).tolist()
    if missing:
        ordinals[set_copy.null_index] = set_copy.null_index
    assert ordinals == list(range(4))

    ar1 = vaex.strings.array(arlist[:2])
    ar2 = vaex.strings.array(arlist[2:])
    oset1 = ordered_set_string(nmaps)
    oset1.update(ar1)
    oset2 = ordered_set_string(nmaps)
    oset2.update(ar2)
    oset1.merge([oset2])
    assert set(oset1.keys()) == set(keys_expected)
    assert set(oset1.key_array().tolist()) == set(keys_expected)

    ar1 = vaex.strings.array(arlist[:1])
    ar2 = vaex.strings.array(arlist[1:])
    oset1 = ordered_set_string(nmaps)
    oset1.update(ar1)
    oset2 = ordered_set_string(nmaps)
    oset2.update(ar2)
    oset1.merge([oset2])
    assert set(oset1.keys()) == set(keys_expected)
    assert set(oset1.key_array().tolist()) == set(keys_expected)


def test_counter_float64(repickle):
    ar = np.arange(3, dtype='f8')
    counter = counter_float64(1)
    counter.update(ar)
    counts = counter.extract()[0]
    assert set(counts.keys()) == {0, 1, 2}
    assert set(counts.values()) == {1, 1, 1}


    counter.update([np.nan, 0])
    counts = counter.extract()[0]
    assert counter.nan_count == 1
    assert counts[0] == 2

    counter.update([np.nan, 0])
    counts = counter.extract()[0]
    assert counter.nan_count == 2
    assert counts[0] == 3

    counter2 = counter_float64(1)
    counter2.update([np.nan, 0, 10])
    assert counter2.nan_count == 1
    counter.merge(counter2)

    counts = counter.extract()[0]
    assert set(counts.keys()) == {0, 1, 2, 10}
    assert counter.nan_count == 3
    assert counts[0] == 4
    assert counts[10] == 1
    assert counts[1] == 1

def test_ordered_set_object():
    s = str("hi there!!")
    ar = np.array([0, 1.5, s, None, s], dtype='O')
    oset = ordered_set_object(-1)
    oset.update(ar)
    keys = np.array(oset.keys())
    assert set(oset.map_ordinal(keys)) == set(list(range(len(keys))))

    ar2 = np.array([np.nan, None, s], dtype='O')
    oset.update(ar2)
    keys = np.array(oset.keys())
    assert set(oset.map_ordinal(keys)) == set(list(range(1, 1 + len(keys))))

def test_counter_object():
    s = str("hi there!!")
    s2 = str("hi there!!2")
    start_ref_count = sys.getrefcount(s)
    start_ref_count2 = sys.getrefcount(s2)

    counter = counter_object(-1)
    ar = np.array([0, 1.5, s, None, s], dtype='O')
    assert sys.getrefcount(s) == start_ref_count+2
    counter.update(ar)
    assert sys.getrefcount(s) == start_ref_count+3
    counts = counter.extract()
    assert sys.getrefcount(s) == start_ref_count+4, 'stored in the dics'
    assert set(counts.keys()) == {0, 1.5, s, None}
    assert counts[0] == 1
    assert counts[1.5] == 1
    assert counts[s] == 2
    assert counts[None] == 1
    del counts
    assert sys.getrefcount(s) == start_ref_count+3, 'released from the dict'



    assert sys.getrefcount(s) == start_ref_count+3
    counter.update(np.array([np.nan, None, s], dtype='O'))
    assert sys.getrefcount(s) == start_ref_count+3
    counts = counter.extract()
    assert counter.nan_count == 1
    assert counts[0] == 1
    assert counts[None] == 2
    assert counts[s] == 3


    counter.update(np.array([np.nan, 0], dtype='O'))
    counts = counter.extract()
    assert counter.nan_count == 2
    assert counts[0] == 2

    counter2 = counter_object(-1)
    ar2 = np.array([np.nan, np.nan, 0, 10, s, s2], dtype='O')
    assert sys.getrefcount(s) == start_ref_count+5
    assert sys.getrefcount(s2) == start_ref_count2+1
    counter2.update(ar2)
    assert sys.getrefcount(s2) == start_ref_count2+2
    assert sys.getrefcount(s) == start_ref_count+6
    assert counter2.nan_count == 2
    counter.merge(counter2)
    assert sys.getrefcount(s2) == start_ref_count2+3
    assert sys.getrefcount(s) == start_ref_count+6
    del counter2
    assert sys.getrefcount(s2) == start_ref_count2+2
    assert sys.getrefcount(s) == start_ref_count+5
    del ar2
    assert sys.getrefcount(s2) == start_ref_count2+1
    assert sys.getrefcount(s) == start_ref_count+4

    counts = counter.extract()
    assert set(counts.keys()) == {0, 1.5, s, s2, None, 10}
    assert counter.nan_count == 4
    assert counts[0] == 3
    assert counts[10] == 1
    del ar
    assert sys.getrefcount(s) == start_ref_count+2
    del counter
    assert sys.getrefcount(s) == start_ref_count+1
    del counts
    assert sys.getrefcount(s) == start_ref_count
    assert sys.getrefcount(s2) == start_ref_count2

def test_index():
    ar1 = np.arange(3, dtype='f8')
    ar2 = np.arange(10, 13, dtype='f8')
    ar = np.concatenate([ar1, ar2])
    index = index_hash_float64(1)
    index.update(ar1, 0)
    assert index.map_index(ar1).tolist() == [0, 1, 2]
    index.update(ar2, 3)
    assert index.map_index(ar).tolist() == [0, 1, 2, 3, 4, 5]

@pytest.mark.parametrize("nmaps", [1, 2, 3])
def test_index_multi(nmaps):
    strings = vaex.strings.array(['aap', 'noot', 'mies'])
    index = index_hash_string(nmaps)
    index.update(strings, 0)
    assert index.map_index(strings).tolist() == [0, 1, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings, 0)] == [[], []]
    assert index.has_duplicates is False
    assert len(index) == 3

    # duplicate that is already present
    strings2 = vaex.strings.array(['aap', 'aap', 'kees', 'mies'])
    index.update(strings2, 3)
    assert index.map_index(strings2).tolist() == [0, 0, 5, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings2, 3)] == [[3, 3, 4, 4, 6], [3, 4, 3, 4, 6]]
    assert index.has_duplicates is True
    assert index.extract() == {'noot': 1, 'aap': [0, 3, 4], 'mies': [2, 6], 'kees': 5}
    assert len(index) == 7

    # duplicate that is not present, and a single one that is already in index
    strings3 = vaex.strings.array(['foo', 'foo', 'mies'])
    index.update(strings3, 7)
    assert index.map_index(strings3).tolist() == [7, 7, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings3, 7)] == [[7, 8, 9, 9], [8, 8, 6, 9]]
    assert index.has_duplicates is True
    assert len(index) == 10

    # same, but now use merge
    strings = vaex.strings.array(['aap', 'noot', 'mies'])
    index = index_hash_string(nmaps)
    index.update(strings, 0)
    assert index.map_index(strings).tolist() == [0, 1, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings, 0)] == [[], []]
    assert index.has_duplicates is False
    assert len(index) == 3

    strings2 = vaex.strings.array(['aap', 'aap', 'kees', 'mies'])
    index2 = index_hash_string(nmaps)
    index2.update(strings2, 3)
    index.merge(index2)
    assert index.map_index(strings2).tolist() == [0, 0, 5, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings2, 3)] == [[3, 3, 4, 4, 6], [3, 4, 3, 4, 6]]
    assert index.has_duplicates is True
    assert len(index) == 7

    strings3 = vaex.strings.array(['foo', 'foo', 'mies'])
    index3 = index_hash_string(nmaps)
    index3.update(strings3, 7)
    index.merge(index3)
    assert index.map_index(strings3).tolist() == [7, 7, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings3, 7)] == [[7, 8, 9, 9], [8, 8, 6, 9]]
    assert index.has_duplicates is True
    assert len(index) == 10


@pytest.mark.parametrize("nmaps", [1, 2, 3])
def test_index_multi_float64(nmaps):
    floats = np.array([1.0, 2.0, 3.0])
    index = index_hash_float64(nmaps)
    index.update(floats, 0)
    assert index.map_index(floats).tolist() == [0, 1, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats, 0)] == [[], []]
    assert index.has_duplicates is False
    assert len(index) == 3

    # duplicate that is already present
    floats2 = np.array([1.0, 1.0, 10.0, 3.0])
    index.update(floats2, 3)
    assert index.map_index(floats2).tolist() == [0, 0, 5, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats2, 3)] == [[3, 3, 4, 4, 6], [3, 4, 3, 4, 6]]
    assert index.has_duplicates is True
    assert len(index) == 7

    # duplicate that is not present, and a single one that is already in index
    floats3 = np.array([99.9, 99.9, 3.0])
    index.update(floats3, 7)
    assert index.map_index(floats3).tolist() == [7, 7, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats3, 7)] == [[7, 8, 9, 9], [8, 8, 6, 9]]
    assert index.has_duplicates is True
    assert len(index) == 10

    # same, but now use merge
    floats = np.array([1.0, 2.0, 3.0])
    index = index_hash_float64(nmaps)
    index.update(floats, 0)
    assert index.map_index(floats).tolist() == [0, 1, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats, 0)] == [[], []]
    assert index.has_duplicates is False
    assert len(index) == 3

    floats2 = np.array([1.0, 1.0, 10.0, 3.0])
    index2 = index_hash_float64(nmaps)
    index2.update(floats2, 3)
    index.merge(index2)
    assert index.map_index(floats2).tolist() == [0, 0, 5, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats2, 3)] == [[3, 3, 4, 4, 6], [3, 4, 3, 4, 6]]
    assert index.has_duplicates is True
    assert len(index) == 7

    floats3 = np.array([99.9, 99.9, 3.0])
    index3 = index_hash_float64(nmaps)
    index3.update(floats3, 7)
    index.merge(index3)
    assert index.map_index(floats3).tolist() == [7, 7, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats3, 7)] == [[7, 8, 9, 9], [8, 8, 6, 9]]
    assert index.has_duplicates is True
    assert len(index) == 10


@pytest.mark.parametrize("nmaps", [1, 2, 3])
def test_index_write(nmaps):
    ints = np.array([1, 2, 3], dtype=np.int32)
    index = index_hash_int32(nmaps)
    index.update(ints, 0)
    assert index.map_index(ints).tolist() == [0, 1, 2]

    indices = np.full(3, -1, dtype=np.int32)
    index.map_index(ints, indices)
    assert indices.tolist() == [0, 1, 2]

    indices = np.full(3, -1, dtype=np.int32)
    mask = np.zeros(3, dtype=bool)
    index.map_index_masked(ints, mask, indices)
    assert indices.tolist() == [0, 1, 2]


def test_set_max_unique(buffer_size):
    df = vaex.from_arrays(x=np.arange(1000))
    with buffer_size(df):
        with pytest.raises(vaex.RowLimitException, match='.* >= 2 .*'):
            df._set('x', limit=2)
        # TODO: this does not happen any more if we have a single set/hashmap
        # with pytest.raises(vaex.RowLimitException, match='.*larger than.*'):
        #     df._set('x', unique_limit=len(df)-1)


@pytest.mark.parametrize("nmaps", [1])#, 2, 3])
def test_string_refs(nmaps):
    strings = vaex.strings.array(['aap', 'noot', 'mies'])
    oset = ordered_set_string(nmaps)
    oset.update(strings, 0)
    strings = oset.key_array()
    refs = sys.getrefcount(strings)
    assert refs == 2
    assert set(strings.tolist()) == {'aap', 'noot', 'mies'}

    set_copy = ordered_set_string(strings, 0, 0, 0, 'fingerprint')
    assert sys.getrefcount(strings) == refs + 1
    strings_copy = oset.key_array()
    # assert sys.getrefcount(strings) == 2
    assert set(strings_copy.tolist()) == {'aap', 'noot', 'mies'}

    # assert index.map_index(strings).tolist() == [0, 1, 2]
    # assert [k.tolist() for k in index.map_index_duplicates(strings, 0)] == [[], []]
    # assert index.has_duplicates is False
    # assert len(index) == 3

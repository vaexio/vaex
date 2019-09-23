from vaex.superutils import *
import vaex.strings
import numpy as np
import sys
import pytest

@pytest.mark.parametrize('counter_cls', [counter_string]) #, counter_stringview])
def test_counter_string(counter_cls):
    strings = vaex.strings.array(['aap', 'noot', 'mies'])
    counter = counter_cls()
    counter.update(strings)
    counts = counter.extract()

    assert counts["aap"] == 1
    assert counts["noot"] == 1
    assert counts["mies"] == 1

    strings2 = vaex.strings.array(['aap', 'n00t'])
    counter.update(strings2)
    counts = counter.extract()
    assert counts["aap"] == 2
    assert counts["noot"] == 1
    assert counts["n00t"] == 1
    assert counts["mies"] == 1


def test_counter_float64():
    ar = np.arange(3, dtype='f8')
    counter = counter_float64()
    oset = ordered_set_float64()
    counter.update(ar)
    oset.update(ar)
    counts = counter.extract()
    assert set(counts.keys()) == {0, 1, 2}
    assert set(counts.values()) == {1, 1, 1}

    keys = np.array(oset.keys())
    assert set(oset.map_ordinal(keys)) == {0, 1, 2}
    assert oset.map_ordinal(keys).dtype.name == 'int8'



    counter.update([np.nan, 0])
    oset.update([np.nan, 0])
    counts = counter.extract()
    assert counter.nan_count == 1
    assert counts[0] == 2

    keys = np.array(oset.keys())
    assert set(oset.map_ordinal(keys)) == {1, 2, 3}

    counter.update([np.nan, 0])
    counts = counter.extract()
    assert counter.nan_count == 2
    assert counts[0] == 3

    counter2 = counter_float64()
    counter2.update([np.nan, 0, 10])
    assert counter2.nan_count == 1
    counter.merge(counter2)

    counts = counter.extract()
    assert set(counts.keys()) == {0, 1, 2, 10}
    assert counter.nan_count == 3
    assert counts[0] == 4
    assert counts[10] == 1
    assert counts[1] == 1

def test_ordered_set_object():
    s = str("hi there!!")
    ar = np.array([0, 1.5, s, None, s], dtype='O')
    oset = ordered_set_object()
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

    counter = counter_object()
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

    counter2 = counter_object()
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
    index = index_hash_float64()
    index.update(ar1, 0)
    assert index.map_index(ar1).tolist() == [0, 1, 2]
    index.update(ar2, 3)
    assert index.map_index(ar).tolist() == [0, 1, 2, 3, 4, 5]

def test_index_multi():
    strings = vaex.strings.array(['aap', 'noot', 'mies'])
    index = index_hash_string()
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
    index = index_hash_string()
    index.update(strings, 0)
    assert index.map_index(strings).tolist() == [0, 1, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings, 0)] == [[], []]
    assert index.has_duplicates is False
    assert len(index) == 3

    strings2 = vaex.strings.array(['aap', 'aap', 'kees', 'mies'])
    index2 = index_hash_string()
    index2.update(strings2, 3)
    index.merge(index2)
    assert index.map_index(strings2).tolist() == [0, 0, 5, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings2, 3)] == [[3, 3, 4, 4, 6], [3, 4, 3, 4, 6]]
    assert index.has_duplicates is True
    assert len(index) == 7

    strings3 = vaex.strings.array(['foo', 'foo', 'mies'])
    index3 = index_hash_string()
    index3.update(strings3, 7)
    index.merge(index3)
    assert index.map_index(strings3).tolist() == [7, 7, 2]
    assert [k.tolist() for k in index.map_index_duplicates(strings3, 7)] == [[7, 8, 9, 9], [8, 8, 6, 9]]
    assert index.has_duplicates is True
    assert len(index) == 10


def test_index_multi_float64():
    floats = np.array([1.0, 2.0, 3.0])
    index = index_hash_float64()
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
    index = index_hash_float64()
    index.update(floats, 0)
    assert index.map_index(floats).tolist() == [0, 1, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats, 0)] == [[], []]
    assert index.has_duplicates is False
    assert len(index) == 3

    floats2 = np.array([1.0, 1.0, 10.0, 3.0])
    index2 = index_hash_float64()
    index2.update(floats2, 3)
    index.merge(index2)
    assert index.map_index(floats2).tolist() == [0, 0, 5, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats2, 3)] == [[3, 3, 4, 4, 6], [3, 4, 3, 4, 6]]
    assert index.has_duplicates is True
    assert len(index) == 7

    floats3 = np.array([99.9, 99.9, 3.0])
    index3 = index_hash_float64()
    index3.update(floats3, 7)
    index.merge(index3)
    assert index.map_index(floats3).tolist() == [7, 7, 2]
    assert [k.tolist() for k in index.map_index_duplicates(floats3, 7)] == [[7, 8, 9, 9], [8, 8, 6, 9]]
    assert index.has_duplicates is True
    assert len(index) == 10

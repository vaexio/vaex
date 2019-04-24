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


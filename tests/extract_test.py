import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time
from common import *

def test_extract(ds_local, ds_trimmed):
    ds = ds_local
    ds_extracted = ds.extract()
    ds_extracted.x.tolist() == ds_trimmed.x.tolist()
    ds_extracted.x.tolist() == np.arange(10.).tolist()
    assert len(ds_extracted) == len(ds_trimmed) == 10
    assert ds_extracted.length_original() == ds_trimmed.length_original() == 10
    assert ds_extracted.length_unfiltered() == ds_trimmed.length_unfiltered() == 10
    assert ds_extracted.filtered is False

    ds_extracted2 = ds_extracted[ds_extracted.x >= 5].extract()
    ds_extracted2.x.tolist() == np.arange(5,10.).tolist()
    assert len(ds_extracted2) == 5
    assert ds_extracted2.length_original() == 5
    assert ds_extracted2.length_unfiltered() == 5
    assert ds_extracted2.filtered is False



def test_thread_safe():
    df = vaex.from_arrays(x=np.arange(int(1e5)))
    dff = df[df.x < 100]

    barrier = Barrier(100)
    def run(_ignore):
        barrier.wait()
        # now we all do the extract at the same time
        dff.extract()
    pool = ThreadPoolExecutor(max_workers=100)
    _values = list(pool.map(run, range(100)))


def test_extract_pre_filtered_and_not_pre_filtered(ds_local) -> None:
    with vaex.cache.off():
        @vaex.delayed
        def add(a, b):
            return a + b

        df = ds_local.copy()
        print("orig is filtered", df.filtered)
        df2 = df[df.y >= 41]
        print("new is filtered", df2.filtered)
        total_promise = add(
            df.sum(df.x, delay=True),
            df2.sum(df2.y + 45, delay=True)
        )
        print("promise is fulfilled", total_promise.isFulfilled)
        print("task 0 is pre_filtered", df.executor.tasks[0].pre_filter)
        print("task 1 is pre_filtered", df.executor.tasks[1].pre_filter)
        df = df.extract()  # has 1 pre_filter task and 1 non_pre_filter task
        df.execute()
        print(total_promise.get())

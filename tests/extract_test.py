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

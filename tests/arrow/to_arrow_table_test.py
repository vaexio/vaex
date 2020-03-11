import numpy as np

import pytest

import vaex


@pytest.mark.parametrize("sample", [100, 1_000, 10_000])
@pytest.mark.parametrize("chunk_size", [5, 10, 30, 100, 500])
def test_to_arrow_table_batches(sample, chunk_size):
    x = np.ma.MaskedArray(data=np.random.randint(low=-5, high=55, size=sample),
                          mask=np.random.choice([True, False], size=sample, p=[0.1, 0.9]),
                          dtype=np.int16)
    df = vaex.from_arrays(x=x)
    df['y'] = (df.x > 15).astype('int')

    def gen():
        features = ['x', 'y']
        for i1, i2, table in df.to_arrow_table(features, chunk_size=chunk_size):
            yield table.to_batches(chunk_size)[0]

    for batch in gen():
        pdf_batch = batch.to_pandas()
        pdf_batch['comp'] = (pdf_batch.x > 15).astype(int)
        y_values = pdf_batch.y.values.copy()
        comp_values = pdf_batch.comp.values.copy().astype(float)
        comp_values[pdf_batch.x.isna()] = np.nan
        np.testing.assert_array_equal(y_values, comp_values)

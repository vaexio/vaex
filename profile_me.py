import vaex

import xarray
df = vaex.open('/home/maartenbreddels/.vaex/benchmarks/numerical_8.hdf5')#[:100_000_000]
df
df.categorize(df.i8_1K, min_value=5, max_value=1000+5)
df.categorize(df.i4_1K, min_value=5, max_value=1000+5)
df.categorize(df.i2_1K, min_value=5, max_value=1000+5)
df.categorize(df.i8_10, min_value=5, max_value=10+5)
df.categorize(df.i4_10, min_value=5, max_value=10+5)
df.categorize(df.i2_10, min_value=5, max_value=10+5)
df.categorize(df.i1_10, min_value=5, max_value=10+5)
df.executor.buffer_size = 1024*1024*2
for i in range(1000):
    #df.count(binby=[df.i8_10])
    #df.binby([df.i8_1K], agg='count')
    #df.binby([df.i8_1K, df.i8_1K, df.i8_10], agg='count')
    df.count(binby=[df.i8_1K, df.i8_1K, df.i8_10])
    #bla = df[df.i8_1K > 1000].binby([df.i8_1K, df.i8_1K, df.i8_10], agg=vaex.agg.count())
    # bla = df.binby([df.i8_1K, df.i8_1K, df.i8_10], agg=vaex.agg.count(selection=df.i8_1K > 1000));

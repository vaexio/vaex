import vaex
df = vaex.open('/Users/maartenbreddels/.vaex/benchmarks/numerical.hdf5')[:100_000_000]
df
df.categorize(df.i8_1K, min_value=5, max_value=1000+5)
df.categorize(df.i4_1K, min_value=5, max_value=1000+5)
df.categorize(df.i2_1K, min_value=5, max_value=1000+5)
df.categorize(df.i8_10, min_value=5, max_value=10+5)
df.categorize(df.i4_10, min_value=5, max_value=10+5)
df.categorize(df.i2_10, min_value=5, max_value=10+5)
df.categorize(df.i1_10, min_value=5, max_value=10+5)

for i in range(100):
    # df.binby([df.i8_1K, df.i8_1K, df.i8_10], agg='count')
    # df.binby([df.i8_10, df.i8_10, df.i8_10], agg='count')
    bla = df[df.i8_1K > 1000].binby([df.i8_1K, df.i8_1K, df.i8_10], agg=vaex.agg.count())
    # bla = df.binby([df.i8_1K, df.i8_1K, df.i8_10], agg=vaex.agg.count(selection=df.i8_1K > 1000));
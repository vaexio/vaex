class DataFrameAccessorViz(object):
    def __init__(self, df):
        self.df = df

    def plot2d(self, *args, **kwargs):
        self.df.plot(*args, **kwargs)

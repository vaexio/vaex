from vaex.utils import _issequence


class Rolling:
    '''Provides rolling window calculations'''
    def __init__(self, df, window, trim, edge, fill_value=None, columns=None):
        self.df = df
        self.window = window
        self.trim = trim
        self.edge = edge
        self.fill_value = fill_value
        self.columns = columns

    def array(self):
        '''Creates an array representing each window'''
        if self.edge == "right":
            return self.df.shift((-self.window, 0), column=self.columns, trim=self.trim, fill_value=self.fill_value)
        elif self.edge == "left":
            return self.df.shift((0, self.window), column=self.columns, trim=self.trim, fill_value=self.fill_value)
        else:
            raise ValueError(f'edge can be "right", "left" or "center", not {self.edge}')

    def sum(self):
        '''Sum all values in the window'''
        df = self.array()
        for column in self.columns:
            df[column] = df[column].sum(axis=1)
        return df

    # def apply(self, f):
    #     df = self.array()
    #     for column in self.columns:
    #         df[column] = df[column].apply(f)
    #     return df

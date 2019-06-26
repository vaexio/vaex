import vaex


def object_type_test():
    df = vaex.read_csv('s3://xdss-public-datasets/demos/titanic.csv')
    assert df['Cabin'].dtype == str

    """
    >> df.dtypes
    ...
    Ticket         <class 'str'> (can we make this into just "str")? 
    Fare                 float64 
    Cabin                 object
    Embarked       <class 'str'>
    index                  int64
    """
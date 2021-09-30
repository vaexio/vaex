from datetime import datetime
import os

from google.cloud import bigquery

import pytest

import vaex.ml
from vaex.contrib.io.gbq import from_query, from_table, to_table


client_project_id = os.getenv('PROJECT_ID')


def test_from_query():
    client_project = client_project_id
    query = '''
    select * from `bigquery-public-data.ml_datasets.iris`
    where species = "virginica"
    '''

    df = from_query(query=query, client_project=client_project)

    assert df.shape == (50, 5)
    assert df.species.unique() == ['virginica']


@pytest.mark.parametrize("export", [None, 'tmp.arrow'])
def test_from_table(export):
    project = 'bigquery-public-data'
    dataset = 'ml_datasets'
    table = 'iris'
    columns = ['species', 'sepal_width', 'petal_width']
    conditions = 'species = "virginica"'

    df = from_table(project=project, dataset=dataset, table=table,
                    columns=columns, condition=conditions,
                    client_project=client_project_id, export=export)
    assert df.shape == (50, 3)
    assert df.species.unique() == ['virginica']


def test_to_table():
    dataset = 'test_dataset'
    table = 'test_upload_table_titanic'

    df = vaex.ml.datasets.load_titanic()

    to_table(df=df, dataset=dataset, table=table)

    # Verify that the table exists
    # That it was created/modified today
    # And that it has the right schema
    client = bigquery.Client()
    table_id = f"vaex-282913.{dataset}.{table}"
    t = client.get_table(table_id)
    assert t.modified.astimezone().date() == datetime.now().date()
    assert len(t.schema) == 14

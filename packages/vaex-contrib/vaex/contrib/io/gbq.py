'''
A module for I/O between Vaex and Google BigQuery.
'''

import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import vaex.utils

google = vaex.utils.optional_import("google", modules=[
    "google.cloud.bigquery",
    "google.cloud.bigquery_storage",
])


def query_google_big_query(query, client_project=None, credentials=None):
    '''Make a query to Google BigQuery and get the result as a Vaex DataFrame.

    :param str query: The SQL query.
    :param str client_project: The ID of the project that executes the query. Will be passed when creating a job. If `None`, falls back to the default inferred from the environment.
    :param credentials: The authorization credentials to attach to requests. See google.auth.credentials.Credentials for more details.
    :rtype: DataFrame
    '''
    client = google.cloud.bigquery.Client(project=client_project, credentials=credentials)
    job = client.query(query=query)
    return vaex.from_arrow_table(job.to_arrow())


def download_google_bigquery_table(project, dataset, table, columns=None, condition=None, export=None, client_project=None, credentials=None):
    '''Download (stream) an entire Google BigQuery table locally.

    :param str project: The Google BigQuery project that owns the table.
    :param str dataset: The dataset the table is part of.
    :param str table: The name of the table
    :param list columns: A list of columns (field names) to download. If None, all columns will be downloaded.
    :param str condition: SQL text filtering statement, similar to a WHERE clause in a query. Aggregates are not supported.
    :param str export: Pass an filename or path to download the table as an Apache Arrow file, and leverage memory mapping. If `None` the DataFrame is in memory.
    :param str client_project: The ID of the project that executes the query. Will be passed when creating a job. If `None`, it will be set with the same value as `project`.
    :param credentials: The authorization credentials to attach to requests. See google.auth.credentials.Credentials for more details.
    :rtype: DataFrame
    '''
    # Instantiate the table path and the reading session
    bq_table = f'projects/{project}/datasets/{dataset}/tables/{table}'
    req_sess = google.cloud.bigquery_storage.types.ReadSession(table=bq_table, data_format=google.cloud.bigquery_storage.types.DataFormat.ARROW)

    # Read options
    req_sess.read_options.selected_fields = columns
    req_sess.read_options.row_restriction = condition

    # Instantiate the reading client
    client = google.cloud.bigquery_storage.BigQueryReadClient(credentials=credentials)

    parent = f'projects/{client_project or project}'
    session = client.create_read_session(parent=parent, read_session=req_sess, max_stream_count=1)
    reader = client.read_rows(session.streams[0].name)

    if export is None:
        arrow_table = reader.to_arrow(session)
        return vaex.from_arrow_table(arrow_table)

    else:
        # We need to get the schema first - Get one RecordsBatch manually to get the schema
        # Get the pages iterator
        pages = reader.rows(session).pages
        # Get the first batch
        first_batch = pages.__next__().to_arrow()
        # Get the schema
        schema = first_batch.schema

        # This does the writing - streams the batches to disk!
        with vaex.file.open(path=export, mode='wb') as sink:
            with pa.RecordBatchStreamWriter(sink, schema) as writer:
                writer.write_batch(first_batch)
                for page in pages:
                    batch = page.to_arrow()
                    writer.write_batch(batch)

        return vaex.open(export)


def upload_to_google_bigquery_table(df, dataset, table, job_config=None, client_project=None, credentials=None, chunk_size=None, progress=None):
    '''Upload a Vaex DataFrame to a Google BigQuery Table.

    Note that the upload creates a temporary parquet file on the local disk, which is then upload to
    Google BigQuery.

    :param DataFrame df: The Vaex DataFrame to be uploaded.
    :param str dataset: The name of the dataset to which the table belongs
    :param str table: The name of the table
    :param job_config: Optional, an instance of google.cloud.bigquery.job.load.LoadJobConfig
    :param str client_project: The ID of the project that executes the query. Will be passed when creating a job. If `None`, falls back to the default inferred from the environment.
    :param credentials: The authorization credentials to attach to requests. See google.auth.credentials.Credentials for more details.
    :param chunk_size: In case the local disk space is limited, export the dataset in chunks.
                       This is considerably slower than a single file upload and it should be avoided.
    :param progress: Valid only if chunk_size is not None. A callable that takes one argument (a floating point value between 0 and 1) indicating the progress, calculations are cancelled when this callable returns False
    '''

    # Instantiate the BigQuery Client
    client = google.cloud.bigquery.Client(project=client_project, credentials=credentials)

    # Confirm configuration of the LoadJobConfig
    if job_config is not None:
        assert isinstance(job_config, google.cloud.bigquery.job.load.LoadJobConfig)
        job_config.source_format = google.cloud.bigquery.SourceFormat.PARQUET
    else:
        job_config = google.cloud.bigquery.LoadJobConfig(source_format=google.cloud.bigquery.SourceFormat.PARQUET)

    # Table to which to upload
    table_bq = f"{dataset}.{table}"

    if chunk_size is None:
        with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
            df.export_parquet(tmp.name)
            with open(tmp.name, "rb") as source_file:
                job = client.load_table_from_file(source_file, table_bq, job_config=job_config)
            job.result()

    else:
        progressbar = vaex.utils.progressbars(progress)
        n_samples = len(df)

        for i1, i2, table in df.to_arrow_table(chunk_size=chunk_size):
            progressbar(i1 / n_samples)
            with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
                pq.write_table(table, tmp.name)
                with open(tmp.name, "rb") as source_file:
                    job = client.load_table_from_file(source_file, table_bq, job_config=job_config)
                job.result()
        progressbar(1.0)

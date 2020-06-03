import logging
from functools import partial

import numpy as np

import tensorflow as tf

import tensorflow_io.arrow as arrow_io
from tensorflow_io.arrow.python.ops.arrow_dataset_ops import arrow_schema_to_tensor_types

import traitlets
import pyarrow as pa

import vaex
import vaex.serialize
from vaex.ml import generate
from vaex.ml import state
from vaex.ml.state import serialize_pickle


class DataFrameAccessorTensorflow(object):
    def __init__(self, ml):
        self.ml = ml
        self.df = self.ml.df

    def arrow_batch_generator(self, column_names=None, chunk_size=1024, parallel=True):
        """Create a generator which yields arrow table batches, to use as datasoure for creating Tensorflow datasets.

        :param features: A list of column names.
        :param target: The dependent or target column, if any.
        :param chunk_size: Number of samples per chunk of data.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.

        Returns:
        :return: generator that yields arrow table batches
        """
        for i1, i2, table in self.df.to_arrow_table(column_names=column_names, chunk_size=chunk_size, parallel=parallel):
            yield table.to_batches(chunk_size)[0]

    def arrow_schema(self, column_names=None):
        return self.df[0:1].to_arrow_table(column_names=column_names, parallel=False).schema

    def tensor_types(self, column_names=None):
        """Returns the output_types, output_shapes tuple"""
        return arrow_schema_to_tensor_types(self.arrow_schema(column_names=column_names))

    def to_dataset(self, features=None, target=None, chunk_size=1024, as_dict=True, parallel=True):
        """Create a tensorflow Dataset object from a DataFrame, via Arrow.

        :param features: A list of column names, default is all but target.
        :param target: The dependent or target column, if any.
        :param chunk_size: Number of samples per chunk of data.
        :param as_dict: If True, the dataset will have the form of dictionary housing the tensors.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.
        This is useful for making inputs directly for tensorflow. If False, the dataset will contain Tensors,
        useful for passing the dataset as a datasource to a Keras model.

        Returns:
        :return ds: A tensorflow Dataset
        """
        if features is None:
            features = self.df.get_column_names()
            if target is not None and target in features:
                features.remove(target)

        if target is not None:
            target = vaex.utils._ensure_list(target)
            target = vaex.utils._ensure_strings_from_expressions(target)
            n_target_cols = len(target)
            column_names = features + target
        else:
            column_names = features

        # Set up the iterator factory
        iterator_factory = partial(self.arrow_batch_generator, **{'column_names': column_names,
                                                                   'chunk_size': chunk_size,
                                                                   'parallel': parallel})
        # get the arrow schema
        output_types, output_shapes = self.tensor_types(column_names)

        # Define the TF dataset
        ds = arrow_io.ArrowStreamDataset.from_record_batches(record_batch_iter=iterator_factory(),
                                                             output_types=output_types,
                                                             output_shapes=output_shapes,
                                                             batch_mode='auto',
                                                             record_batch_iter_factory=iterator_factory)

        # Reshape the data into the appropriate format
        if as_dict:
            if target is not None:
                if n_target_cols == 1:
                    ds = ds.map(lambda *tensors: (dict(zip(features, tensors[:-1])), tensors[-1]))
                else:
                    ds = ds.map(lambda *tensors: (dict(zip(features, tensors[:-n_target_cols])),
                                                  tf.stack(tensors[-n_target_cols:], axis=1)))
            else:
                ds = ds.map(lambda *tensors: (dict(zip(features, tensors))))
        else:
            if target is not None:
                if n_target_cols == 1:
                    ds = ds.map(lambda *tensors: (tf.stack(tensors[:-1], axis=1), tensors[-1]))
                else:
                    ds = ds.map(lambda *tensors: (tf.stack(tensors[:-n_target_cols], axis=1),
                                                  tf.stack(tensors[-n_target_cols:], axis=1)))
            else:
                ds = ds.map(lambda *tensors: (tf.stack(tensors, axis=1)))

        return ds

    def make_input_function(self, features, target=None, chunk_size=1024, repeat=None, shuffle=False, parallel=True):
        """Create a tensorflow Dataset object from a DataFrame, via Arrow.

        :param features: A list of column names.
        :param target: The dependent or target column, if any.
        :param chunk_size: Number of samples per chunk of data.
        :param repeat: If not None, repeat the dataset as many times as specified.
        :param shuffle: If True, the elements of the dataset are randomly shuffled. If shuffle is True and repeat is not None,
        the dataset will first be repeated, and the entire repeated dataset shuffled.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.

        Returns:
        :return ds: A tensorflow Dataset
        """
        if repeat is not None:
            assert (isinstance(repeat, int)) & (repeat > 0), 'The "repeat" arg must be a positive integer larger larger than 0.'
            shuffle_buffer_size = chunk_size * repeat
        else:
            shuffle_buffer_size = chunk_size

        def tf_input_function():
            ds = self.to_dataset(features=features, target=target, chunk_size=chunk_size, parallel=True)
            if repeat is not None:
                ds = ds.repeat(repeat)
            if shuffle:
                ds = ds.shuffle(shuffle_buffer_size)

            return ds

        return tf_input_function

    def to_keras_generator(self, features, target=None, chunk_size=1024, parallel=True, verbose=True):
        """Return a batch generator suitable as a Keras datasource.  Note that the generator is infinite, i.e. it loops
        continuously ovder the data. Thus you need to specify the "steps_per_epoch" arg when fitting a keras model,
        the "validation_steps" when using it for validation, and "steps" when calling the "predict" method of a keras model.

        :param features: A list of column names.
        :param target: The dependent or target column or a list of columns, if any.
        :param chunk_size: Number of samples per chunk of data. This can be thought of as the batch size.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.
        :parallel verbose: If True, show an info on the recommended "steps_per_epoch"
        based on the total number of samples and "chunk_size".
        """
        if verbose:
            steps_per_epoch = np.ceil(len(self.df) / chunk_size)
            logging.info(f'Recommended "steps_per_epoch" arg: {steps_per_epoch}')

        target = target

        def _generator(features, target, chunk_size, parallel):
            if target is not None:
                target = vaex.utils._ensure_list(target)
                target = vaex.utils._ensure_strings_from_expressions(target)
                n_target_cols = len(target)
                column_names = features + target
            else:
                column_names = features

            while True:
                if target is not None:
                    for i1, i2, chunks, in self.df.evaluate_iterator(column_names, chunk_size=chunk_size, parallel=parallel):
                        X = np.array(chunks[:-n_target_cols]).T
                        y = np.array(chunks[-n_target_cols:], copy=False).T
                        yield (X, y)

                else:
                    for i1, i2, chunks, in self.df.evaluate_iterator(column_names, chunk_size=chunk_size, parallel=parallel):
                        X = np.array(chunks).T
                        yield (X, )

        return _generator(features, target, chunk_size, parallel)


@vaex.serialize.register
@generate.register
class Model(state.HasState):
    model = traitlets.Any(default_value=None, allow_none=True, help='A tensorflow estimator with a `.fit_predict` method.').tag()
    features = traitlets.List(traitlets.Unicode(), help='List of features to use.')
    # Maybe should look at https://github.com/vaexio/vaex/pull/553
    features_ = traitlets.List(traitlets.Unicode(), help='List of output feature names').tag(output=True)
    target = traitlets.Unicode(allow_none=False, help='The name of the target column.')
    output_types = traitlets.Any()
    output_shapes = traitlets.Any()

    def fit(self, df, repeat=10, shuffle=True):
        raise RuntimeError("Fit not implemented, for flexibility reasons it is better you manually run it")

    def init(self, df):
        '''This will set the output features, types and shapes based on sample data from df'''
        # we need 1 sample
        sample_fn = df[0:1].ml.tensorflow.make_input_function(features=self.features)
        sample_pred_result = list(self.model.predict(sample_fn, yield_single_examples=False))[0]
        self.features_ = list(sample_pred_result.keys())
        self.output_types, self.output_shapes = df.ml.tensorflow.tensor_types(self.features)

    def transform(self, df):
        df = df.copy()
        lazy_function = df.add_function('df_estimator_function', self, unique=True)
        # TODO: this causes an evaluation of the tensorflow model multiple times
        # if we support arrow, we can return a struct
        for feature_output in self.features_:
            expression = lazy_function(str(repr(feature_output)), *self.features)
            df.add_virtual_column(feature_output, expression, unique=False)
        return df

    def __call__(self, feature_output_name, *args):
        arrow_table = pa.Table.from_arrays(args, self.features)
        batches = arrow_table.to_batches(len(arrow_table))

        def tf_input_function():
            def iterator_factory():
                yield batches[0]
            ds = arrow_io.ArrowStreamDataset.from_record_batches(record_batch_iter=iterator_factory(),
                                                                output_types=self.output_types,
                                                                output_shapes=self.output_shapes,
                                                                batch_mode='auto',
                                                                record_batch_iter_factory=iterator_factory)
            ds = ds.map(lambda *tensors: (dict(zip(self.features, tensors))))
            return ds
        prediction = list(self.model.predict(tf_input_function, yield_single_examples=False))[0]
        return prediction[feature_output_name]

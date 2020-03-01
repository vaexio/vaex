import logging
from functools import partial

import numpy as np

import tensorflow as tf

import tensorflow_io.arrow as arrow_io
from tensorflow_io.arrow.python.ops.arrow_dataset_ops import arrow_schema_to_tensor_types

import vaex


class DataFrameAccessorTensorflow(object):
    def __init__(self, ml):
        self.ml = ml
        self.df = self.ml.df

    def arrow_batch_generator(self, column_names, chunk_size=1024, parallel=True):
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

    @staticmethod
    def _get_batch_arrow_schema(arrow_batch):
        """Get the schema from a arrow batch table."""
        output_types, output_shapes = arrow_schema_to_tensor_types(arrow_batch.schema)
        return output_types, output_shapes

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
        output_types, output_shapes = self._get_batch_arrow_schema(next(iterator_factory()))

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

def docsubst(f):
    if f.__doc__:
        f.__doc__ = f.__doc__.format(**_doc_snippets)
    return f


_doc_snippets = {}
_doc_snippets["expression"] = "expression or list of expressions, e.g. df.x, 'x', or ['x, 'y']"
_doc_snippets["expression_one"] = "expression in the form of a string, e.g. 'x' or 'x+y' or vaex expression object, e.g. df.x or df.x+df.y "
_doc_snippets["expression_single"] = "if previous argument is not a list, this argument should be given"
_doc_snippets["binby"] = "List of expressions for constructing a binned grid"
_doc_snippets["limits"] = """description for the min and max values for the expressions, e.g. 'minmax' (default), '99.7%', [0, 10], or a list of, e.g. [[0, 10], [0, 20], 'minmax']"""
_doc_snippets["shape"] = """shape for the array where the statistic is calculated on, if only an integer is given, it is used for all dimensions, e.g. shape=128, shape=[128, 256]"""
_doc_snippets["percentile_limits"] = """description for the min and max values to use for the cumulative histogram, should currently only be 'minmax'"""
_doc_snippets["percentile_shape"] = """shape for the array where the cumulative histogram is calculated on, integer type"""
_doc_snippets["selection"] = """Name of selection to use (or True for the 'default'), or all the data (when selection is None or False), or a list of selections"""
_doc_snippets["selection1"] = """Name of selection to use (or True for the 'default'), or all the data (when selection is None or False)"""
_doc_snippets["delay"] = """Do not return the result, but a proxy for asychronous calculations (currently only for internal use)"""
_doc_snippets["progress"] = """True to display a progress bar, or a callable that takes one argument (a floating point value between 0 and 1) indicating the progress, calculations are cancelled when this callable returns False"""
_doc_snippets["expression_limits"] = _doc_snippets["expression"]
_doc_snippets["grid"] = """If grid is given, instead if compuation a statistic given by what, use this Nd-numpy array instead, this is often useful when a custom computation/statistic is calculated, but you still want to use the plotting machinery."""
_doc_snippets["edges"] = """Currently for internal use only (it includes nan's and values outside the limits at borders, nan and 0, smaller than at 1, and larger at -1"""

_doc_snippets["healpix_expression"] = """Expression which maps to a healpix index, for the Gaia catalogue this is for instance 'source_id/34359738368', other catalogues may simply have a healpix column."""
_doc_snippets["healpix_max_level"] = """The healpix level associated to the healpix_expression, for Gaia this is 12"""
_doc_snippets["healpix_level"] = """The healpix level to use for the binning, this defines the size of the first dimension of the grid."""

_doc_snippets["return_stat_scalar"] = """Numpy array with the given shape, or a scalar when no binby argument is given, with the statistic"""
_doc_snippets["return_limits"] = """List in the form [[xmin, xmax], [ymin, ymax], .... ,[zmin, zmax]] or [xmin, xmax] when expression is not a list"""
_doc_snippets["cov_matrix"] = """List all convariance values as a double list of expressions, or "full" to guess all entries (which gives an error when values are not found), or "auto" to guess, but allow for missing values"""
_doc_snippets['propagate_uncertainties'] = """If true, will propagate errors for the new virtual columns, see :meth:`propagate_uncertainties` for details"""
_doc_snippets['note_copy'] = '.. note:: Note that no copy of the underlying data is made, only a view/reference is made.'
_doc_snippets['note_filter'] = '.. note:: Note that filtering will be ignored (since they may change), you may want to consider running :meth:`extract` first.'
_doc_snippets['inplace'] = 'If True, make modifications to self, otherwise return a new DataFrame'
_doc_snippets['return_shallow_copy'] = 'Returns a new DataFrame with a shallow copy/view of the underlying data'
_doc_snippets['chunk_size'] = 'Return an iterator with cuts of the object in lenght of this size'
_doc_snippets['chunk_size_export'] = 'Number of rows to be written to disk in a single iteration'
_doc_snippets['evaluate_parallel'] = 'Evaluate the (virtual) columns in parallel'
_doc_snippets['array_type'] = 'Type of output array, possible values are None (keep as is), "numpy" (ndarray), "xarray" for a xarray.DataArray, or "list"/"python" for a Python list'
_doc_snippets['ascii'] = 'Transform only ascii characters (usually faster).'
_doc_snippets['fs_options'] = 'see :func:`vaex.open` e.g. for S3 {"profile": "myproject"}'
_doc_snippets['fs'] = 'Pass a file system object directly, see :func:`vaex.open`'
_doc_snippets['random_state'] = 'Sets a seed or `RandomState` for reproducability. When `None`, a random seed it chosen.'
_doc_snippets['trim'] = 'Trim off begin or end of dataframe to avoid missing values'
_doc_snippets['limit'] = 'Limit the amount of results'
_doc_snippets['limit_raise'] = 'Raise :exc:`vaex.RowLimitException` when limit is exceeded, or return at maximum \'limit\' amount of results.'
_doc_snippets['dropmissing'] = 'Drop rows with missing values'
_doc_snippets['dropnan'] = 'Drop rows with NaN values'
_doc_snippets['dropna'] = 'Drop rows with Not Available (NA) values (NaN or missing values).'

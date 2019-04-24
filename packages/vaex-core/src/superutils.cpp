#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>

namespace py = pybind11;

#define custom_isnan(value) (!(value==value))

namespace vaex {
    void init_hash_primitives(py::module &);
    void init_hash_string(py::module &);
    void init_hash_object(py::module &);
}

PYBIND11_MODULE(superutils, m) {
    _import_array();

    m.doc() = "fast utils";

    vaex::init_hash_primitives(m);
    vaex::init_hash_string(m);
    vaex::init_hash_object(m);
}

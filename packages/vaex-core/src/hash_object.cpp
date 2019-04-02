#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include "hash.hpp"
#include <math.h>

namespace py = pybind11;

namespace std {
    template<>
    struct hash<PyObject*> {
        size_t operator()(const PyObject *const &o) const {
            return PyObject_Hash((PyObject*)o);
        }
    };
}

namespace vaex {

struct CompareObjects
{
    bool operator()(const PyObject*const &a, const PyObject*const &b) const
    {
        return PyObject_RichCompareBool((PyObject*)a, (PyObject*)b, Py_EQ) == 1;
    }
};

struct CompareObjectsPyBind
{
    bool operator()(const py::object &a, const py::object &b) const
    {
        return PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_EQ) == 1;
    }
};

class counter_object {
public:
    typedef PyObject* value_type;
    typedef PyObject* storage_type;
    counter_object() : counts(0), nan_count(0) {};
    ~counter_object();
    void update(py::buffer object_array);
    void merge(const counter_object&);
    py::object extract();
    hashmap<storage_type, int64_t, std::hash<storage_type>, CompareObjects> map;
    bool has_nan;
    int64_t counts;
    int64_t nan_count;
};

counter_object::~counter_object() {
    for(auto el : map) {
        PyObject* key = el.first;
        Py_DECREF(key);
    }    
}

void counter_object::update(py::buffer object_array) {
    py::buffer_info info = object_array.request();
    if(info.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    // TODO: check dtype/format
    int64_t length = info.shape[0];
    PyObject** array = (PyObject**)info.ptr;
    for(int64_t i = 0; i < length; i++) {
        value_type value = array[i];
        if(PyFloat_Check(value) && isnan(PyFloat_AsDouble(value))) {
            nan_count++;
        } else {
            auto search = map.find(value);
            auto end = map.end();
            if(search == end) {
                Py_IncRef(value);
                map.emplace(value, 1);
            } else {
                set_second(search, search->second + 1);
            }
        }
    }
}
void counter_object::merge(const counter_object & other) {
    for (auto & elem : other.map) {
        const value_type& value = elem.first;
        auto search = map.find(value);
        auto end = map.end();
        if(search == end) {
            map.emplace(elem);
        } else {
            set_second(search, search->second + elem.second);
        }
    }
    nan_count += other.nan_count;
}

/*
// pybind11 does not seem to be able to work with std::map<py::object, int64_t> ?
// could be an issue with the custom comare
std::map<py::object, int64_t, CompareObjectsPyBind> counter_object::extract() {
    std::map<py::object, int64_t, CompareObjectsPyBind> m;
    for(auto el : map) {
        storage_type storage_value = el.first;
        // value_type value = reinterpret_cast<value_type&>(storage_value); 
        py::object value = py::reinterpret_steal<py::object>(storage_value);
        m[value] = el.second;
    }
    return m;
}
/**/
py::object counter_object::extract() {
    PyObject* dict = PyDict_New();
    for(auto el : map) {
        PyObject* count = PyLong_FromLongLong(el.second);
        PyDict_SetItem(dict, el.first, count);
        Py_DECREF(count);
    }
    return  py::reinterpret_steal<py::object>(dict);
}

void init_hash_object(py::module &m) {
    typedef counter_object counter_type;
    std::string countername = "counter_object";
    py::class_<counter_type>(m, countername.c_str())
        .def(py::init<>())
        .def("update", &counter_type::update)
        .def("merge", &counter_type::merge)
        .def("extract", &counter_type::extract)
        .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
    ;
    // std::string hashsetname = "hashset_" + name;
    // typedef hashset<T> hashset_type;
    // py::class_<hashset_type>(m, hashsetname.c_str())
    //     .def(py::init<>())
    //     .def("update", &hashset_type::update)
    //     .def("merge", &hashset_type::merge)
    //     .def("extract", &hashset_type::extract)
    // ;

}
}
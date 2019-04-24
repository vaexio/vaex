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


template<class Derived, class T, class A=T>
class hash_base {
public:
    using value_type = T;
    using storage_type = A;
    hash_base() : count(0), nan_count(0), null_count(0) {} ;
    virtual ~hash_base() {
        for(auto el : map) {
            PyObject* key = el.first;
            Py_DECREF(key);
        }    
    }

    void update(py::buffer object_array) {
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
                    static_cast<Derived&>(*this).add(value);
                } else {
                    static_cast<Derived&>(*this).add(search, value);
                }
            }
        }
    }
    void update_with_mask(py::buffer object_array, py::array_t<bool>& masks) {
        auto m = masks.template unchecked<1>();
        py::buffer_info info = object_array.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d byte buffer");
        }
        // TODO: check dtype/format
        int64_t length = info.shape[0];
        assert(m.size() == length);
        PyObject** array = (PyObject**)info.ptr;
        for(int64_t i = 0; i < length; i++) {
            value_type value = array[i];
            if(m[i]) {
                null_count++;
            } else if(PyFloat_Check(value) && isnan(PyFloat_AsDouble(value))) {
                nan_count++;
            } else {
                auto search = map.find(value);
                auto end = map.end();
                if(search == end) {
                    static_cast<Derived&>(*this).add(value);
                } else {
                    static_cast<Derived&>(*this).add(search, value);
                }
            }
        }
    }
    py::object keys() {
        PyObject* list = PyList_New(this->map.size());
        size_t index = 0;
        for(auto el : map) {
            Py_IncRef(el.first);
            PyList_SetItem(list, index++, el.first);
        }
        return  py::reinterpret_steal<py::object>(list);
    }
    py::object extract() {
        PyObject* dict = PyDict_New();
        for(auto el : map) {
            PyObject* count = PyLong_FromLongLong(el.second);
            PyDict_SetItem(dict, el.first, count);
            Py_DECREF(count);
        }
        return  py::reinterpret_steal<py::object>(dict);
    }
    hashmap<storage_type, int64_t, std::hash<storage_type>, CompareObjects> map;
    int64_t count;
    int64_t nan_count;
    int64_t null_count;
};

template<class T=PyObject*, class A=T>
class counter : public hash_base<counter<T, A>, T, A> {
public:
    using typename hash_base<counter<T, A>, T, A>::value_type;
    using typename hash_base<counter<T, A>, T, A>::storage_type;

    void add(storage_type& storage_value) {
        Py_IncRef(storage_value);
        this->map.emplace(storage_value, 1);
    }
    template<class Bucket>
    void add(Bucket& bucket, storage_type& storage_value) {
        set_second(bucket, bucket->second + 1);
    }
    void merge(const counter & other) {
        for (auto & elem : other.map) {
            const value_type& value = elem.first;
            auto search = this->map.find(value);
            auto end = this->map.end();
            if(search == end) {
                Py_IncRef(value);
                this->map.emplace(elem);
            } else {
                set_second(search, search->second + elem.second);
            }
        }
        this->nan_count += other.nan_count;
        this->null_count += other.null_count;
    }
};

template<class T=PyObject*>
class ordered_set : public hash_base<ordered_set<T>, T, T> {
public:
    using typename hash_base<ordered_set<T>, T, T>::value_type;
    using typename hash_base<ordered_set<T>,T, T>::storage_type;
    py::array_t<int64_t> map_ordinal(py::buffer object_array) {
        py::buffer_info info = object_array.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d byte buffer");
        }
        int64_t size = info.shape[0];
        PyObject** array = (PyObject**)info.ptr;

        // TODO: check dtype/format

        py::array_t<int64_t> result(size);
        auto output = result.template mutable_unchecked<1>();
        // assert(m.size() == size);
        // null and nan map to 0 and 1, and move the index up
        int64_t offset = (this->null_count > 0 ? 1 : 0) + (this->nan_count > 0 ? 1 : 0);
        for(int64_t i = 0; i < size; i++) {
            value_type value = array[i];
            if(PyFloat_Check(value) && isnan(PyFloat_AsDouble(value))) {
                output(i) = 0;
            } else {
                auto search = this->map.find(value);
                auto end = this->map.end();
                if(search == end) {
                    output(i) = -1;
                } else {
                    output(i) = search->second + offset;
                }
            }
        }
        return result;
    }
    py::array_t<int64_t> map_ordinal_with_mask(py::buffer object_array, py::array_t<bool>& masks) {
        auto m = masks.template unchecked<1>();
        py::buffer_info info = object_array.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d byte buffer");
        }
        // TODO: check dtype/format
        int64_t size = info.shape[0];
        py::array_t<int64_t> result(size);
        auto output = result.template mutable_unchecked<1>();
        assert(m.size() == size);
        PyObject** array = (PyObject**)info.ptr;
        // null and nan map to 0 and 1, and move the index up
        int64_t offset = (this->null_count > 0 ? 1 : 0) + (this->nan_count > 0 ? 1 : 0);
        for(int64_t i = 0; i < size; i++) {
            value_type value = array[i];
            if(m[i]) {
                output(i) = offset;
            } else if(PyFloat_Check(value) && isnan(PyFloat_AsDouble(value))) {
                output(i) = 0;
            } else {
                auto search = this->map.find(value);
                auto end = this->map.end();
                if(search == end) {
                    output(i) = -1;
                } else {
                    output(i) = search->second + offset;
                }
            }
        }
        return result;
    }
    void add(storage_type& storage_value) {
        Py_IncRef(storage_value);
        this->map.emplace(storage_value, this->count);
        this->count++;
    }
    template<class Bucket>
    void add(Bucket& position, storage_type& storage_value) {
        // we can do nothing here
    }
    void merge(const ordered_set & other) {
        for (auto & elem : other.map) {
            const value_type& value = elem.first;
            auto search = this->map.find(value);
            auto end = this->map.end();
            if(search == end) {
                Py_IncRef(value);
                this->map.emplace(value, this->count);
                this->count++;
            } else {
                // if already in, it's fine
            }
        }
        this->nan_count += other.nan_count;
        this->null_count += other.null_count;
    }
    py::object keys() {
        PyObject* list = PyList_New(this->map.size());
        for(auto el : this->map) {
            Py_IncRef(el.first);
            PyList_SetItem(list, el.second, el.first);
        }
        return  py::reinterpret_steal<py::object>(list);
    }
};


void init_hash_object(py::module &m) {
    {
        typedef counter<> Type;
        std::string countername = "counter_object";
        py::class_<Type>(m, countername.c_str())
            .def(py::init<>())
            .def("update", &Type::update)
            .def("update", &Type::update_with_mask)
            .def("merge", &Type::merge)
            .def("extract", &Type::extract)
            .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
        ;
    }
    {
        std::string ordered_setname = "ordered_set_object";
        typedef ordered_set<> Type;
        py::class_<Type>(m, ordered_setname.c_str())
            .def(py::init<>())
            .def("update", &Type::update)
            .def("update", &Type::update_with_mask)
            .def("merge", &Type::merge)
            .def("extract", &Type::extract)
            .def("keys", &Type::keys)
            .def("map_ordinal", &Type::map_ordinal)
            .def("map_ordinal", &Type::map_ordinal_with_mask)
            .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
            .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
            .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
        ;

    }

}
}

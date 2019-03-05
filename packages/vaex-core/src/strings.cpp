#include <iostream>
#include <string>
#include <regex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "string_utils.hpp"
#include <Python.h>

#ifdef USE_XPRESSIVE
#include <boost/xpressive/xpressive.hpp>
namespace xp = boost::xpressive;
#endif


namespace py = pybind11;

class StringSequence {
    public:
    StringSequence(size_t length) : length(length) {
    }
    virtual ~StringSequence() {
    }
    virtual string_view view(size_t i) const = 0;
    virtual const std::string get(size_t i) const = 0;
    py::object search(const std::string pattern, bool regex) {
        py::array_t<bool> matches(length);
        auto m = matches.mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            if(regex) {
                #ifdef USE_XPRESSIVE
                    xp::sregex rex = xp::sregex::compile(pattern);
                #else
                    std::regex rex(pattern);
                #endif
                for(size_t i = 0; i < length; i++) {
                    #ifdef USE_XPRESSIVE
                        std::string str = get(i);
                        bool match = xp::regex_search(str, rex);
                    #else
                        auto str = view(i);
                        bool match = regex_search(str, rex);
                    #endif
                    m(i) = match;
                }
            } else {
                for(size_t i = 0; i < length; i++) {
                    auto str = view(i);
                    m(i) = str.find(pattern) != std::string::npos;
                }
            }
        }
        return matches;
    }
    py::object get(size_t start, size_t end) {
        size_t count = end - start;
        npy_intp shape[1];
        shape[0] = count;
        PyObject* array = PyArray_SimpleNew(1, shape, NPY_OBJECT);
        PyArray_XDECREF((PyArrayObject*)array);
        PyObject **ptr = (PyObject**)PyArray_DATA((PyArrayObject*)array);
        for(size_t i = start; i < end; i++) {
            if( (i < 0) || (i > length) ) {
                throw std::runtime_error("out of bounds i2");
            }
            string_view str = view(i);
            ptr[i - start] = PyUnicode_FromStringAndSize(str.begin(), str.length());;
        }
        py::handle h = array;
        return py::reinterpret_steal<py::object>(h);
    }
    size_t length;
};

class StringList : public StringSequence {
public:
    StringList(const char *bytes, size_t byte_length, int32_t *indices, size_t string_count, size_t offset)
     : StringSequence(string_count), bytes(bytes), byte_length(byte_length), indices(indices), offset(offset) {
    }
    void print() {
        // std::cout << get();
    }
    void _check(int64_t i) const {
        if( (i < 0) || (i > length) ) {
            throw std::runtime_error("string index out of bounds");
        }
        int32_t i1 = indices[i] - offset;
        int32_t i2 = indices[i+1] - offset;
        int32_t count = i2 - i1;
        if( (i1 < 0) || (i1 > byte_length)) {
            throw std::runtime_error("out of bounds i1");
        }
        if( (i2 < 0) || (i2 > byte_length) ) {
            printf("i1/i2 = %ld %ld length %ld i=%ld", i1, i2, byte_length, i);
            throw std::runtime_error("out of bounds i2");
        }

    }
    // const std::string get() const {
    //     return std::string(bytes, 0, byte_length);
    // }
    virtual string_view view(size_t i) const {
        _check(i);
        int32_t start = indices[i] - offset;
        int32_t end = indices[i+1] - offset;
        int32_t count = end - start;
        return string_view(bytes + start, count);
    }
    virtual const std::string get(size_t i) const {
        _check(i);
        int32_t start = indices[i] - offset;
        int32_t end = indices[i+1] - offset;
        int32_t count = end - start;
        return std::string(bytes, start, count);
    }

public:
    const char* bytes;
    size_t byte_length;
    const int32_t* indices;
    size_t offset;
};

int add(int i, int j) {
    return i + j + 1;
}

const char* empty = "";

class StringArray : public StringSequence {
public:
    StringArray(PyObject** object_array, size_t length) : StringSequence(length) {
        #if PY_MAJOR_VERSION == 2
            utf8_objects= (PyObject**)malloc(length * sizeof(void*));
        #endif
        strings = (char**)malloc(length * sizeof(void*));
        sizes = (Py_ssize_t*)malloc(length * sizeof(Py_ssize_t));
        for(size_t i = 0; i < length; i++) {
            #if PY_MAJOR_VERSION == 3
                if(PyUnicode_CheckExact(object_array[i])) {
                    strings[i] = PyUnicode_AsUTF8AndSize(object_array[i], &sizes[i]);
                } else {
                    strings[i] = 0;
                }
            #else
                if(PyUnicode_CheckExact(object_array[i])) {
                    // if unicode, first convert to utf8
                    utf8_objects[i] = PyUnicode_AsUTF8String(object_array[i]);
                    sizes[i] = PyString_Size(utf8_objects[i]);
                    strings[i] = PyString_AsString(utf8_objects[i]);
                } else if(PyString_CheckExact(object_array[i])) {
                    // otherwise directly use
                    utf8_objects[i] = 0;
                    sizes[i] = PyString_Size(object_array[i]);
                    strings[i] = PyString_AsString(object_array[i]);
                }
            #endif
        }
    }
    ~StringArray() {
        free(strings);
        free(sizes);
        #if PY_MAJOR_VERSION == 2
            for(size_t i = 0; i < length; i++) {
                if(utf8_objects[i])
                    Py_XDECREF(utf8_objects[i]);
            }
            utf8_objects= (PyObject**)malloc(length * sizeof(void*));
        #endif
    }
    virtual string_view view(size_t i) const {
        if(strings[i] == 0) {
            return string_view(empty);
        }
        return string_view(strings[i], sizes[i]);
    }
    virtual const std::string get(size_t i) const {
        if(strings[i] == 0) {
            return std::string(empty);
        }
        return std::string(strings[i], sizes[i]);
    }
    #if PY_MAJOR_VERSION == 2
        PyObject** utf8_objects;
    #endif
    char** strings;
    Py_ssize_t* sizes;
};

PYBIND11_MODULE(strings, m) {
    _import_array();
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
    py::class_<StringSequence> string_sequence(m, "StringSequence");
    string_sequence
        .def("search", &StringSequence::search, "Tests if strings contains pattern", py::arg("pattern"), py::arg("regex"))//, py::call_guard<py::gil_scoped_release>())
        .def("get", (py::object (StringSequence::*)(size_t, size_t))&StringSequence::get, py::return_value_policy::take_ownership)
    ;
    py::class_<StringList>(m, "StringList", string_sequence)
        .def(py::init([](py::buffer bytes, py::array_t<int32_t, py::array::c_style>& indices, size_t string_count, size_t offset) {
                // bytes.inc_ref();
                // indices.inc_ref();
                py::buffer_info bytes_info = bytes.request();
                py::buffer_info indices_info = indices.request();
                // std::cout << indices;
                if(bytes_info.ndim != 1) {
                    throw std::runtime_error("Expected a 1d byte buffer");
                }
                if(indices_info.ndim != 1) {
                    throw std::runtime_error("Expected a 1d indices buffer");
                }
                return std::unique_ptr<StringList>(
                    new StringList((char*)bytes_info.ptr, bytes_info.shape[0],
                                   (int32_t*)indices_info.ptr, string_count, offset
                                  )
                );
            })
        )
        .def("get", (const std::string (StringList::*)(int64_t))&StringList::get)
        // bug? we have to add this again
        .def("get", (py::object (StringSequence::*)(size_t, size_t))&StringSequence::get, py::return_value_policy::take_ownership)
        // .def("print", &StringList::print)
        // .def_property_readonly("bytes", [](const StringList &sl) {
        //         return py::bytes(sl.bytes, sl.byte_length);
        //     }
        // )
        // .def("__repr__",
        //     [](const StringList &sl) {
        //         return "<vaex.strings.StringList buffer='" + sl.get() + "'>";
        //     }
        // )
        ;
    py::class_<StringArray>(m, "StringArray", string_sequence)
        .def(py::init([](py::buffer string_array) {
                py::buffer_info info = string_array.request();
                if(info.ndim != 1) {
                    throw std::runtime_error("Expected a 1d byte buffer");
                }
                // std::cout << info.format << " format" << std::endl;
                return std::unique_ptr<StringArray>(
                    new StringArray((PyObject**)info.ptr, info.shape[0]));
            })
        )
        .def("get", (const std::string (StringArray::*)(int64_t))&StringArray::get)
        // bug? we have to add this again
        .def("get", (py::object (StringSequence::*)(size_t, size_t))&StringSequence::get, py::return_value_policy::take_ownership)
        // .def("print", &StringList::print)
        // .def_property_readonly("bytes", [](const StringList &sl) {
        //         return py::bytes(sl.bytes, sl.byte_length);
        //     }
        // )
        // .def("__repr__",
        //     [](const StringList &sl) {
        //         return "<vaex.strings.StringList buffer='" + sl.get() + "'>";
        //     }
        // )
        ;
}
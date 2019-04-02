#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include "hash.hpp"


#include <nonstd/string_view.hpp>
#include <string>
typedef nonstd::string_view string_view;
typedef std::string string;

namespace py = pybind11;


class StringSequence {
    public:
    virtual ~StringSequence() {
    }
    virtual string_view view(size_t i) const = 0;
    virtual const std::string get(size_t i) const = 0;
    size_t length;
    uint8_t* null_bitmap;
    int64_t null_offset;
};

namespace vaex {

class counter_string {
public:
    typedef string value_type;
    typedef string storage_type;
    counter_string() : counts(0), nan_count(0) {}  ;
    void update(StringSequence* strings);
    void merge(const counter_string&);
    std::map<value_type, int64_t> extract();
    hashmap<storage_type, int64_t> map;
    bool has_nan;
    int64_t counts;
    int64_t nan_count;
};

void counter_string::update(StringSequence* strings) {
    py::gil_scoped_release gil;
    int64_t length = strings->length;
    for(int64_t i = 0; i < length; i++) {
        value_type value = strings->get(i);
        auto search = map.find(value);
        auto end = map.end();
        if(search == end) {
            map.emplace(value, 1);
        } else {
            (*search).second += 1;
        }
    }
}

void counter_string::merge(const counter_string & other) {
    py::gil_scoped_release gil;
    for (auto & elem : other.map) {
            const value_type& value = elem.first;
            auto search = map.find(value);
            auto end = map.end();
            if(search == end) {
                map.emplace(elem);
            } else {
                (*search).second += elem.second;
            }
    }
}

std::map<string, int64_t> counter_string::extract() {
    std::map<value_type, int64_t> m = std::map<value_type, int64_t>(this->map.begin(), this->map.end());
    return m;
}

void init_hash_string(py::module &m) {
    typedef counter_string counter_type;
    std::string countername = "counter_string";
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

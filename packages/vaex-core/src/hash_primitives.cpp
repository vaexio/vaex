#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include "hash.hpp"


namespace py = pybind11;
#define custom_isnan(value) (!(value==value))
namespace vaex {

template<class T, class A=T>
class counter {
public:
    typedef T value_type;
    typedef A storage_type;
    counter() : counts(0), nan_count(0) {}  ;
    void update(py::array_t<value_type>&);
    void merge(const counter&);
    std::map<value_type, int64_t> extract();
    hashmap<storage_type, int64_t> map;
    bool has_nan;
    int64_t counts;
    int64_t nan_count;
};

template<class T,class A>
void counter<T, A>::update(py::array_t<value_type>& values) {
    py::gil_scoped_release gil;
    auto m = values.template unchecked<1>();
    int64_t size = m.size();
    for(int64_t i = 0; i < size; i++) {
            value_type value = m(i);
            if(custom_isnan(value)) {
                nan_count++;
            } else {
                storage_type storage_value = *((storage_type*)(&value));
                auto search = map.find(storage_value);
                auto end = map.end();
                if(search == end) {
                    map.emplace(storage_value, 1);
                } else {
                    set_second(search, search->second + 1);
                }
            }
    }
}

template<class T, class A>
void counter<T, A>::merge(const counter & other) {
    py::gil_scoped_release gil;
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

template<class C, class A>
std::map<C, int64_t> counter<C, A>::extract() {
    std::map<value_type, int64_t> m;
    for(auto el : map) {
        storage_type storage_value = el.first;
        value_type value = *((value_type*)(&storage_value));
        m[value] = el.second;

    }
    return m;
}


template<class T>
class hashset {
public:
    typedef T value_type;
    hashset() : counts(0) {}  ;
    void update(py::array_t<value_type>&);
    void merge(const hashset&);
    std::set<value_type> extract();
    // ska::unordered_set<value_type> set;
    // tsl::sparse_set<value_type> set;
    tsl::hopscotch_set<value_type> set;
    bool has_nan;
    int counts;
};

template<class T>
void hashset<T>::update(py::array_t<value_type>& values) {
    py::gil_scoped_release gil;
    auto m = values.template unchecked<1>();
    int64_t size = m.size();
    for(int64_t i = 0; i < size; i++) {
        value_type value = m(i);
        set.emplace(value);
    }
}

template<class T>
void hashset<T>::merge(const hashset & other) {
    py::gil_scoped_release gil;
    for (auto & elem : other.set) {
        set.emplace(elem);
    }
}

template<class C>
std::set<C> hashset<C>::extract() {
    std::set<value_type> m = std::set<value_type>(this->set.begin(), this->set.end());
    return m;
}

template<class T, class S=T, class M>
void init_hash(M m, std::string name) {
    typedef counter<T, S> counter_type;
    std::string countername = "counter_" + name;
    py::class_<counter_type>(m, countername.c_str())
        .def(py::init<>())
        .def("update", &counter_type::update)
        .def("merge", &counter_type::merge)
        .def("extract", &counter_type::extract)
        .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
    ;
    std::string hashsetname = "hashset_" + name;
    typedef hashset<T> hashset_type;
    py::class_<hashset_type>(m, hashsetname.c_str())
        .def(py::init<>())
        .def("update", &hashset_type::update)
        .def("merge", &hashset_type::merge)
        .def("extract", &hashset_type::extract)
    ;
}



void init_hash_primitives(py::module &m) {
    init_hash<int64_t>(m, "int64");
    init_hash<uint64_t>(m, "uint64");
    init_hash<int32_t>(m, "int32");
    init_hash<uint32_t>(m, "uint32");
    init_hash<int16_t>(m, "int16");
    init_hash<uint16_t>(m, "uint16");
    init_hash<int8_t>(m, "int8");
    init_hash<uint8_t>(m, "uint8");
    init_hash<bool>(m, "bool");
    init_hash<float>(m, "float32");
    init_hash<double, uint64_t>(m, "float64");

}
}

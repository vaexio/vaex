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

template<class Derived, class T, class A=T>
class hash_base {
public:
    using value_type = T;
    using storage_type = A;
    hash_base() : count(0), nan_count(0), null_count(0) {}  ;
    void update(py::array_t<value_type>& values) {
        py::gil_scoped_release gil;
        auto ar = values.template unchecked<1>();
        int64_t size = ar.size();
        for(int64_t i = 0; i < size; i++) {
                value_type value = ar(i);
                if(custom_isnan(value)) {
                    this->nan_count++;
                } else {
                    storage_type storage_value = *((storage_type*)(&value));
                    auto search = this->map.find(storage_value);
                    auto end = this->map.end();
                    if(search == end) {
                        static_cast<Derived&>(*this).add(storage_value);
                    } else {
                        static_cast<Derived&>(*this).add(search, storage_value);
                    }
                }
        }
    }
    void update_with_mask(py::array_t<value_type>& values, py::array_t<bool>& masks) {
        py::gil_scoped_release gil;
        auto ar = values.template unchecked<1>();
        auto m = masks.template unchecked<1>();
        assert(m.size() == ar.size());
        int64_t size = ar.size();
        for(int64_t i = 0; i < size; i++) {
                value_type value = ar(i);
                if(m[i]) {
                    this->null_count++;
                } else if(custom_isnan(value)) {
                    this->nan_count++;
                } else {
                    storage_type storage_value = *((storage_type*)(&value));
                    auto search = this->map.find(storage_value);
                    auto end = this->map.end();
                    if(search == end) {
                        static_cast<Derived&>(*this).add(storage_value);
                    } else {
                        static_cast<Derived&>(*this).add(search, storage_value);
                    }
                }
        }
    }
    std::vector<value_type> keys() {
        std::vector<value_type> v;
        for(auto el : this->map) {
            storage_type storage_value = el.first;
            value_type value = *((value_type*)(&storage_value));
            v.push_back(value);

        }
        return v;
    }
    std::map<value_type, int64_t> extract() {
        std::map<value_type, int64_t> m;
        for(auto el : this->map) {
            storage_type storage_value = el.first;
            value_type value = *((value_type*)(&storage_value));
            m[value] = el.second;

        }
        return m;

    }
    hashmap<storage_type, int64_t> map;
    int64_t count;
    int64_t nan_count;
    int64_t null_count;
};

template<class T, class A=T>
class counter : public hash_base<counter<T, A>, T, A> {
public:
    using typename hash_base<counter<T, A>, T, A>::value_type;
    using typename hash_base<counter<T, A>, T, A>::storage_type;

    void add(storage_type& storage_value) {
        this->map.emplace(storage_value, 1);
    }
    template<class Bucket>
    void add(Bucket& bucket, storage_type& storage_value) {
        set_second(bucket, bucket->second + 1);
    }
    void merge(const counter & other) {
        py::gil_scoped_release gil;
        for (auto & elem : other.map) {
            const value_type& value = elem.first;
            auto search = this->map.find(value);
            auto end = this->map.end();
            if(search == end) {
                this->map.emplace(elem);
            } else {
                set_second(search, search->second + elem.second);
            }
        }
        this->nan_count += other.nan_count;
        this->null_count += other.null_count;
    }
};

template<class T>
class ordered_set : public hash_base<ordered_set<T>, T> {
public:
    using typename hash_base<ordered_set<T>, T, T>::value_type;
    using typename hash_base<ordered_set<T>,T, T>::storage_type;

    static ordered_set* create(std::map<value_type, int64_t> dict, int64_t count, int64_t nan_count, int64_t null_count) {
        ordered_set* set = new ordered_set;
        for(auto el : dict) {
            storage_type storage_value = el.first;
            set->map.emplace(storage_value, el.second);
        }
        set->count = count;
        set->nan_count = nan_count;
        set->null_count = null_count;
        return set;
    }

    py::object map_ordinal(py::array_t<value_type>& values) {
        size_t size = this->map.size() + (this->null_count > 0 ? 1 : 0) + (this->nan_count > 0 ? 1 : 0);
        // TODO: apply this pattern of various return types to the other set types
        if(size < (1u<<7u)) {
            return this->template _map_ordinal<int8_t>(values);
        } else
        if(size < (1u<<15u)) {
            return this->template _map_ordinal<int16_t>(values);
        } else
        if(size < (1u<<31u)) {
            return this->template _map_ordinal<int32_t>(values);
        } else {
            return this->template _map_ordinal<int64_t>(values);
        }
    }
    template<class OutputType>
    py::array_t<OutputType> _map_ordinal(py::array_t<value_type>& values) {
        int64_t size = values.size();
        py::array_t<OutputType> result(size);
        auto input = values.template unchecked<1>();
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        // null and nan map to 0 and 1, and move the index up
        OutputType offset = (this->null_count > 0 ? 1 : 0) + (this->nan_count > 0 ? 1 : 0);
        for(int64_t i = 0; i < size; i++) {
            const value_type& value = input(i);
            if(custom_isnan(value)) {
                output(i) = 0;
                assert(this->nan_count > 0);
            } else {
                storage_type storage_value = *((storage_type*)(&value));
                auto search = this->map.find(storage_value);
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
        this->map.emplace(storage_value, this->count);
        this->count++;
    }
    template<class Bucket>
    void add(Bucket& position, storage_type& storage_value) {
        // we can do nothing here
    }
    void merge(const ordered_set & other) {
        py::gil_scoped_release gil;
        for (auto & elem : other.map) {
            const value_type& value = elem.first;
            auto search = this->map.find(value);
            auto end = this->map.end();
            if(search == end) {
                this->map.emplace(value, this->count);
                this->count++;
            } else {
                // if already in, it's fine
            }
        }
        this->nan_count += other.nan_count;
        this->null_count += other.null_count;
    }
    std::vector<value_type> keys() {
        std::vector<value_type> v(this->map.size());
        for(auto el : this->map) {
            storage_type storage_value = el.first;
            value_type value = *((value_type*)(&storage_value));
            v[el.second] = value;

        }
        return v;
    }
};


template<class T, class S=T, class M>
void init_hash(M m, std::string name) {
    typedef counter<T, S> counter_type;
    std::string countername = "counter_" + name;
    py::class_<counter_type>(m, countername.c_str())
        .def(py::init<>())
        .def("update", &counter_type::update)
        .def("update", &counter_type::update_with_mask)
        .def("merge", &counter_type::merge)
        .def("extract", &counter_type::extract)
        .def("keys", &counter_type::keys)
        .def_property_readonly("count", [](const counter_type &c) { return c.count; })
        .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
        .def_property_readonly("null_count", [](const counter_type &c) { return c.null_count; })
        .def_property_readonly("has_nan", [](const counter_type &c) { return c.nan_count > 0; })
        .def_property_readonly("has_null", [](const counter_type &c) { return c.null_count > 0; })
    ;
    std::string ordered_setname = "ordered_set_" + name;
    typedef ordered_set<T> Type;
    py::class_<Type>(m, ordered_setname.c_str())
        .def(py::init<>())
        .def(py::init(&Type::create))
        .def("update", &Type::update)
        .def("update", &Type::update_with_mask)
        .def("merge", &Type::merge)
        .def("extract", &Type::extract)
        .def("keys", &Type::keys)
        .def("map_ordinal", &Type::map_ordinal)
        .def_property_readonly("count", [](const Type &c) { return c.count; })
        .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
        .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
        .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
        .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
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

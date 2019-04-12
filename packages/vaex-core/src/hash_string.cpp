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


inline bool _is_null(uint8_t* null_bitmap, size_t i) {
    if(null_bitmap) {
        size_t byte_index = i / 8;
        size_t bit_index = (i % 8);
        return (null_bitmap[byte_index] & (1 << bit_index)) == 0;
    } else {
        return false;
    }
}


class StringSequence {
    public:
    virtual ~StringSequence() {
    }
    virtual string_view view(size_t i) const = 0;
    virtual const std::string get(size_t i) const = 0;
    virtual size_t byte_size() const = 0;
    virtual bool is_null(size_t i) const {
        return _is_null(null_bitmap, i + null_offset);
    }
    size_t length;
    uint8_t* null_bitmap;
    int64_t null_offset;
};

namespace vaex {

template<class Derived, class T, class A=T>
class hash_base {
public:
    using value_type = T;
    using storage_type = A;
    hash_base() : count(0), nan_count(0), null_count(0) {}  ;
    void update(StringSequence* strings) {
        py::gil_scoped_release gil;
        int64_t size = strings->length;
        for(int64_t i = 0; i < size; i++) {
                if(strings->is_null(i)) {
                    null_count++;
                } else {
                    auto value = strings->get(i);
                    auto search = this->map.find(value);
                    auto end = this->map.end();
                    if(search == end) {
                        static_cast<Derived&>(*this).add(value);
                    } else {
                        static_cast<Derived&>(*this).add(search, value);
                    }
                }
        }
    }
    void update_with_mask(StringSequence* strings, py::array_t<bool>& masks) {
        py::gil_scoped_release gil;
        int64_t size = strings->length;
        auto m = masks.template unchecked<1>();
        assert(m.size() == size);
        for(int64_t i = 0; i < size; i++) {
                if(strings->is_null(i) || m[i]) {
                    null_count++;
                } else {
                    auto value = strings->get(i);
                    auto search = this->map.find(value);
                    auto end = this->map.end();
                    if(search == end) {
                        static_cast<Derived&>(*this).add(value);
                    } else {
                        static_cast<Derived&>(*this).add(search, value);
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

template<class T=string, class A=T>
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

template<class T=string>
class ordered_set : public hash_base<ordered_set<T>, T, T> {
public:
    using typename hash_base<ordered_set<T>, T, T>::value_type;
    using typename hash_base<ordered_set<T>,T, T>::storage_type;
    py::array_t<int64_t> map_ordinal(StringSequence* strings) {
        int64_t size = strings->length;
        py::array_t<int64_t> result(size);
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        // null and nan map to 0 and 1, and move the index up
        int64_t offset = (this->null_count > 0 ? 1 : 0);
        for(int64_t i = 0; i < size; i++) {
            if(strings->is_null(i)) {
                output(i) = 0;
                assert(this->null_count > 0);
            } else {
                const value_type& value = strings->get(i);
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
};


void init_hash_string(py::module &m) {
    {
        typedef counter<> counter_type;
        std::string countername = "counter_string";
        py::class_<counter_type>(m, countername.c_str())
            .def(py::init<>())
            .def("update", &counter_type::update)
            .def("merge", &counter_type::merge)
            .def("extract", &counter_type::extract)
            .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const counter_type &c) { return c.null_count; })
        ;
    }
    {
        std::string ordered_setname = "ordered_set_string";
        typedef ordered_set<> Type;
        py::class_<Type>(m, ordered_setname.c_str())
            .def(py::init<>())
            .def("update", &Type::update)
            .def("update", &Type::update_with_mask)
            .def("merge", &Type::merge)
            .def("extract", &Type::extract)
            .def("keys", &Type::keys)
            .def("map_ordinal", &Type::map_ordinal)
            .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
            .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
            .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
        ;

    }
}
}

#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include "hash.hpp"
#include "superstring.hpp"

namespace py = pybind11;

// #define VAEX_HASH_STRING_SIMPLE

namespace vaex {

// cannot get stringview to work with msvc
// struct equal_string {
//     using is_transparent = void;

//     bool operator()(const string& str, const string_view& strview) const {
//         return str == strview;
//     }

//     bool operator()(const string_view& strview1, const string_view& strview2) const {
//         return strview1 == strview2;
//     }

//     bool operator()(const string_view strview, const string& str) const {
//         return strview == str;
//     }

//     bool operator()(const string& str1, const string& str2) const {
//         return str1 == str2;
//     }
// };

// struct hash_string {
//     std::size_t operator()(const string& str) const {
//         #ifdef VAEX_HASH_STRING_SIMPLE
//             unsigned int hash = 1;
//             const char *s = str.data();
//             const char *end = s + str.size();
//             while(s != end) {
//                 hash = hash * 101  +  *s++;
//             }
//             return hash;
//         #else
//             return std::hash<string>()(str);
//         #endif
//     }

//     std::size_t operator()(const string_view str_view) const {
//         #ifdef VAEX_HASH_STRING_SIMPLE
//             unsigned int hash = 1;
//             const char *s = str_view.data();
//             const char *end = s + str_view.size();
//             while(s != end) {
//                 hash = hash * 101  +  *s++;
//             }
//             return hash;
//         #else
//             return std::hash<string_view>()(str_view);
//         #endif
//     }
// };

template<class Derived, class T, class A=T, class V=T>
class hash_base {
public:
    using value_type = T;
    using storage_type = A;
    using storage_type_view = V;
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

    // hashmap<storage_type, int64_t, hash_string, equal_string> map;
    hashmap<storage_type, int64_t> map;
    int64_t count;
    int64_t nan_count;
    int64_t null_count;
};

template<class T=string, class A=T, class V=string>
class counter : public hash_base<counter<T, A>, T, A, V> {
public:
    using typename hash_base<counter<T, A>, T, A, V>::value_type;
    using typename hash_base<counter<T, A>, T, A, V>::storage_type;
    using typename hash_base<counter<T, A>, T, A, V>::storage_type_view;

    void add(storage_type_view& storage_view_value) {
        this->map.emplace(storage_view_value, 1);
    }
    template<class Bucket>
    void add(Bucket& bucket, storage_type_view& storage_view_value) {
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

template<class T=string, class V=string>
class ordered_set : public hash_base<ordered_set<T>, T, T, V> {
public:
    using typename hash_base<ordered_set<T>, T, T, V>::value_type;
    using typename hash_base<ordered_set<T>, T, T, V>::storage_type;
    using typename hash_base<ordered_set<T>, T, T, V>::storage_type_view;

    static ordered_set* create(std::map<value_type, int64_t> dict, int64_t count, int64_t nan_count, int64_t null_count) {
        ordered_set* set = new ordered_set;
        for(auto el : dict) {
            storage_type storage_value = el.first;
            set->map.emplace(storage_value, el.second);
            // value_type value = *((value_type*)(&storage_value));
            // m[value] = el.second;
        }
        set->count = count;
        set->nan_count = nan_count;
        set->null_count = null_count;
        return set;
    }

    py::object map_ordinal(StringSequence* strings) {
        size_t size = this->map.size() + (this->null_count > 0 ? 1 : 0);
        if(size < (1u<<7u)) {
            return this->template _map_ordinal<int8_t>(strings);
        } else
        if(size < (1u<<15u)) {
            return this->template _map_ordinal<int16_t>(strings);
        } else
        if(size < (1u<<31u)) {
            return this->template _map_ordinal<int32_t>(strings);
        } else {
            return this->template _map_ordinal<int64_t>(strings);
        }
    }
    template<class OutputType>
    py::array_t<OutputType> _map_ordinal(StringSequence* strings) {
        int64_t size = strings->length;
        py::array_t<OutputType> result(size);
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        // null and nan map to 0 and 1, and move the index up
        OutputType offset = (this->null_count > 0 ? 1 : 0);
        if(strings->has_null()) {
            for(int64_t i = 0; i < size; i++) {
                if(strings->is_null(i)) {
                    output(i) = 0;
                    assert(this->null_count > 0);
                } else {
                    const storage_type_view& value = strings->get(i);
                    auto search = this->map.find(value);
                    auto end = this->map.end();
                    if(search == end) {
                        output(i) = -1;
                    } else {
                        output(i) = search->second + offset;
                    }
                }
            }
        } else {
            for(int64_t i = 0; i < size; i++) {
                const storage_type_view& value = strings->get(i);
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

    void add(storage_type_view& storage_value) {
        this->map.emplace(storage_value, this->count);
        this->count++;
    }

    template<class Bucket>
    void add(Bucket& position, storage_type_view& storage_view_value) {
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
    std::vector<string> keys() {
        std::vector<string> v(this->map.size());
        for(auto el : this->map) {
            storage_type storage_value = el.first;
            value_type value = *((value_type*)(&storage_value));
            string export_value(value);
            v[el.second] = export_value;

        }
        return v;
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
            .def_property_readonly("count", [](const counter_type &c) { return c.count; })
            .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const counter_type &c) { return c.null_count; })
        ;
    }
    // {
    //     typedef counter<string_view, string_view, string_view> counter_type;
    //     std::string countername = "counter_stringview";
    //     py::class_<counter_type>(m, countername.c_str())
    //         .def(py::init<>())
    //         .def("update", &counter_type::update)
    //         .def("merge", &counter_type::merge)
    //         .def("extract", &counter_type::extract)
    //         .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
    //         .def_property_readonly("null_count", [](const counter_type &c) { return c.null_count; })
    //     ;
    // }
    {
        std::string ordered_setname = "ordered_set_string";
        typedef ordered_set<> Type;
        py::class_<Type>(m, ordered_setname.c_str())
            .def(py::init<>())
            .def(py::init(&Type::create))
            .def("update", &Type::update)
            // .def("update", &Type::update_with_mask)
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
    // {
    //     std::string ordered_setname = "ordered_set_stringview";
    //     typedef ordered_set<string_view, string_view> Type;
    //     py::class_<Type>(m, ordered_setname.c_str())
    //         .def(py::init<>())
    //         .def("update", &Type::update)
    //         // .def("update", &Type::update_with_mask)
    //         .def("merge", &Type::merge)
    //         .def("extract", &Type::extract)
    //         .def("keys", &Type::keys)
    //         .def("map_ordinal", &Type::map_ordinal)
    //         .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
    //         .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
    //         .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
    //         .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
    //     ;
    // }
}
}

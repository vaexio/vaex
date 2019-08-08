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
    void update(py::array_t<value_type>& values, int64_t start_index=0) {
        py::gil_scoped_release gil;
        auto ar = values.template unchecked<1>();
        int64_t size = ar.size();
        for(int64_t i = 0; i < size; i++) {
                value_type value = ar(i);
                if(custom_isnan(value)) {
                    this->nan_count++;
                    static_cast<Derived&>(*this).add_nan(start_index + i);
                } else {
                    storage_type storage_value = *((storage_type*)(&value));
                    auto search = this->map.find(storage_value);
                    auto end = this->map.end();
                    if(search == end) {
                        static_cast<Derived&>(*this).add(storage_value, start_index + i);
                    } else {
                        static_cast<Derived&>(*this).add(search, storage_value, start_index + i);
                    }
                }
        }
    }
    void update_with_mask(py::array_t<value_type>& values, py::array_t<bool>& masks, int64_t start_index=0) {
        py::gil_scoped_release gil;
        auto ar = values.template unchecked<1>();
        auto m = masks.template unchecked<1>();
        assert(m.size() == ar.size());
        int64_t size = ar.size();
        for(int64_t i = 0; i < size; i++) {
                value_type value = ar(i);
                if(m[i]) {
                    this->null_count++;
                    static_cast<Derived&>(*this).add_missing(start_index + i);
                } else if(custom_isnan(value)) {
                    this->nan_count++;
                    static_cast<Derived&>(*this).add_nan(start_index + i);
                } else {
                    storage_type storage_value = *((storage_type*)(&value));
                    auto search = this->map.find(storage_value);
                    auto end = this->map.end();
                    if(search == end) {
                        static_cast<Derived&>(*this).add(storage_value, start_index + i);
                    } else {
                        static_cast<Derived&>(*this).add(search, storage_value, start_index + i);
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

    void add_missing(int64_t index) {
    }
    void add_nan(int64_t index) {
    }
    void add(storage_type& storage_value, int64_t index) {
        this->map.emplace(storage_value, 1);
    }
    template<class Bucket>
    void add(Bucket& bucket, storage_type& storage_value, int64_t index) {
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
    void add_nan(int64_t index) {
    }
    void add_missing(int64_t index) {
    }
    void add(storage_type& storage_value, int64_t index) {
        this->map.emplace(storage_value, this->count);
        this->count++;
    }
    template<class Bucket>
    void add(Bucket& position, storage_type& storage_value, int64_t index) {
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

template<class T>
class index_hash : public hash_base<index_hash<T>, T> {
public:
    using typename hash_base<index_hash<T>, T, T>::value_type;
    using typename hash_base<index_hash<T>,T, T>::storage_type;
    typedef hashmap<storage_type, std::vector<int64_t>> MultiMap;

    py::array_t<int64_t> map_index(py::array_t<value_type>& values) {
        int64_t size = values.size();
        py::array_t<int64_t> result(size);
        auto input = values.template unchecked<1>();
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        for(int64_t i = 0; i < size; i++) {
            const value_type& value = input(i);
            if(custom_isnan(value)) {
                output(i) = nan_index;
                assert(this->nan_count > 0);
            } else {
                storage_type storage_value = *((storage_type*)(&value));
                auto search = this->map.find(storage_value);
                auto end = this->map.end();
                if(search == end) {
                    output(i) = -1;
                } else {
                    output(i) = search->second;
                }
            }
        }
        return result;
    }
    py::array_t<int64_t> map_index_with_mask(py::array_t<value_type>& values, py::array_t<uint8_t>& mask) {
        int64_t size = values.size();
        assert(values.size() == mask.size());
        py::array_t<int64_t> result(size);
        auto input = values.template unchecked<1>();
        auto input_mask = mask.template unchecked<1>();
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        for(int64_t i = 0; i < size; i++) {
            const value_type& value = input(i);
            if(custom_isnan(value)) {
                output(i) = nan_index;
                assert(this->nan_count > 0);
            } else
            if(input_mask(i) == 1) {
                output(i) = missing_index;
                assert(this->nan_count > 0);
            } else {
                storage_type storage_value = *((storage_type*)(&value));
                auto search = this->map.find(storage_value);
                auto end = this->map.end();
                if(search == end) {
                    output(i) = -1;
                } else {
                    output(i) = search->second;
                }
            }
        }
        return result;
    }

    std::tuple<py::array_t<int64_t>, py::array_t<int64_t>> map_index_duplicates_with_mask(py::array_t<value_type>& values, py::array_t<uint8_t>& mask, int64_t start_index) {
        std::vector<typename MultiMap::value_type> found; // should this be a reference to the value_type?
        std::vector<int64_t> indices;
        size_t size = values.size();
        size_t size_output = 0;

        auto input = values.template unchecked<1>();
        auto input_mask = mask.template unchecked<1>();

        const auto end = this->multimap.end(); // we don't modify the multimap, so keep this const
        {
            py::gil_scoped_release gil;
            for(size_t i = 0; i < size; i++) {
                const value_type& value = input(i);
                if(custom_isnan(value)) {
                } else
                if(input_mask(i) == 1) {
                } else {
                    auto search = this->multimap.find(value);
                    if(search != end) {
                        found.push_back(*search);
                        size_output += search->second.size();
                        indices.insert(indices.end(), search->second.size(), start_index+i);
                    }
                }
            }
        }

        py::array_t<int64_t> result(size_output);
        py::array_t<int64_t> indices_array(size_output);
        auto output = result.template mutable_unchecked<1>();
        auto output_indices = indices_array.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        // int64_t offset = 0;
        size_t index = 0;

        std::copy(indices.begin(), indices.end(), &output_indices(0));

        for(auto el : found) {
            std::vector<int64_t>& indices = el.second;
            for(int64_t i : indices) {
                output(index++) = i;
            }
        }
        return std::make_tuple(indices_array, result);
    }

    std::tuple<py::array_t<int64_t>, py::array_t<int64_t>> map_index_duplicates(py::array_t<value_type>& values, int64_t start_index) {
        std::vector<typename MultiMap::value_type> found; // should this be a reference to the value_type?
        std::vector<int64_t> indices;
        size_t size = values.size();
        size_t size_output = 0;

        auto input = values.template unchecked<1>();

        const auto end = this->multimap.end(); // we don't modify the multimap, so keep this const
        {
            py::gil_scoped_release gil;
            for(size_t i = 0; i < size; i++) {
                const value_type& value = input(i);
                if(custom_isnan(value)) {
                } else {
                    auto search = this->multimap.find(value);
                    if(search != end) {
                        found.push_back(*search);
                        size_output += search->second.size();
                        indices.insert(indices.end(), search->second.size(), start_index+i);
                    }
                }
            }
        }

        py::array_t<int64_t> result(size_output);
        py::array_t<int64_t> indices_array(size_output);
        auto output = result.template mutable_unchecked<1>();
        auto output_indices = indices_array.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        // int64_t offset = 0;
        size_t index = 0;

        std::copy(indices.begin(), indices.end(), &output_indices(0));

        for(auto el : found) {
            std::vector<int64_t>& indices = el.second;
            for(int64_t i : indices) {
                output(index++) = i;
            }
        }
        return std::make_tuple(indices_array, result);
    }

    void add_nan(int64_t index) {
        this->nan_index = index;
    }
    void add_missing(int64_t index) {
        this->missing_index = index;
    }
    void add(storage_type& storage_value, int64_t index) {
        this->map.emplace(storage_value, index);
        this->count++;
    }
    template<class Bucket>
    void add(Bucket& position, storage_type& storage_value, int64_t index) {
        // we found a duplicate
        multimap[position->first].push_back(index);
        has_duplicates = true;
        this->count++;
    }
    void merge(const index_hash & other) {
        py::gil_scoped_release gil;
        for (auto & elem : other.map) {
            const value_type& value = elem.first;
            auto search = this->map.find(value);
            auto end = this->map.end();
            if(search == end) {
                this->map.emplace(value, elem.second);
            } else {
                // if already in, add it to the multimap
                multimap[elem.first].push_back(elem.second);
            }
            this->count++;
        }
        this->nan_count += other.nan_count;
        this->null_count += other.null_count;
        for(auto el : other.multimap) {
            std::vector<int64_t>& source = el.second;

            storage_type& storage_value = el.first;
            // const value_type& value = elem.first;
            auto search = this->map.find(storage_value);
            auto end = this->map.end();
            if(search == end) {
                // we have a duplicate that is not in the current map, so we insert the first element
                this->map.emplace(storage_value, source[0]);
                if(source.size() > 1) {
                    std::vector<int64_t>& target = this->multimap[storage_value];
                    target.insert(target.end(), source.begin()+1, source.end());
                }
            } else {
                // easy case, just merge the vectors
                std::vector<int64_t>& target = this->multimap[storage_value];
                target.insert(target.end(), source.begin(), source.end());
            }
            this->count += source.size();
        }
        has_duplicates = has_duplicates || other.has_duplicates;
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
    int64_t missing_index;
    int64_t nan_index;
    MultiMap multimap; // this stores only the duplicates
    bool has_duplicates;
};

template<class T, class S=T, class M>
void init_hash(M m, std::string name) {
    typedef counter<T, S> counter_type;
    std::string countername = "counter_" + name;
    py::class_<counter_type>(m, countername.c_str())
        .def(py::init<>())
        .def("update", &counter_type::update, "add values", py::arg("values"), py::arg("start_index") = 0)
        .def("update", &counter_type::update_with_mask, "add masked values", py::arg("values"), py::arg("masks"), py::arg("start_index") = 0)
        .def("merge", &counter_type::merge)
        .def("extract", &counter_type::extract)
        .def("keys", &counter_type::keys)
        .def_property_readonly("count", [](const counter_type &c) { return c.count; })
        .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
        .def_property_readonly("null_count", [](const counter_type &c) { return c.null_count; })
        .def_property_readonly("has_nan", [](const counter_type &c) { return c.nan_count > 0; })
        .def_property_readonly("has_null", [](const counter_type &c) { return c.null_count > 0; })
    ;
    {
        std::string ordered_setname = "ordered_set_" + name;
        typedef ordered_set<T> Type;
        py::class_<Type>(m, ordered_setname.c_str())
            .def(py::init<>())
            .def(py::init(&Type::create))
            .def("update", &Type::update, "add values", py::arg("values"), py::arg("start_index") = 0)
            .def("update", &Type::update_with_mask, "add masked values", py::arg("values"), py::arg("masks"), py::arg("start_index") = 0)
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
    {
        std::string index_hashname = "index_hash_" + name;
        typedef index_hash<T> Type;
        py::class_<Type>(m, index_hashname.c_str())
            .def(py::init<>())
            .def("update", &Type::update)
            .def("update", &Type::update_with_mask)
            .def("merge", &Type::merge)
            .def("extract", &Type::extract)
            .def("keys", &Type::keys)
            .def("map_index", &Type::map_index)
            .def("map_index", &Type::map_index_with_mask)
            .def("map_index_duplicates", &Type::map_index_duplicates)
            .def("__len__", [](const Type &c) { return c.count + (c.null_count > 0) + (c.nan_count > 0); })
            .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
            .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
            .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
            .def_property_readonly("has_duplicates", [](const Type &c) { return c.has_duplicates; })
        ;
    }
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

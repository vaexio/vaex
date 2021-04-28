#include "hash.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>

namespace py = pybind11;
#define custom_isnan(value) (!(value==value))


namespace vaex {

template<class Key, class Value>
using hashmap_primitive = hashmap<Key, Value>;

template<class Key, class Value>
using hashmap_primitive_pg = hashmap_pg<Key, Value>;

template<class Derived, class T, template<typename, typename> typename Hashmap>
class hash_base {
public:
    using value_type = T;
    hash_base() : count(0), nan_count(0), null_count(0) {}  ;
    void reserve(int64_t count_) {
        py::gil_scoped_release gil;
        map.reserve(count_);
    }
    void update(py::array_t<value_type>& values, int64_t start_index=0) {
        py::gil_scoped_release gil;
        auto ar = values.template unchecked<1>();
        int64_t size = ar.size();
        for(int64_t i = 0; i < size; i++) {
                value_type value = ar(i);
                if(custom_isnan(value)) {
                    // this->nan_count++;
                    // static_cast<Derived&>(*this).add_nan(start_index + i);
                    update1_nan(start_index + i);
                } else {
                    update1(value, start_index + i);
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
                    // this->null_count++;
                    // static_cast<Derived&>(*this).add_missing(start_index + i);
                    update1_null(start_index + i);
                } else if(custom_isnan(value)) {
                    // this->nan_count++;
                    // static_cast<Derived&>(*this).add_nan(start_index + i);
                    update1_nan(start_index + i);
                } else {
                    update1(value, start_index + i);
                }
        }
    }
    void update1(value_type& value, int64_t index=0) {
        auto search = this->map.find(value);
        auto end = this->map.end();
        if(search == end) {
            static_cast<Derived&>(*this).add(value, index);
        } else {
            static_cast<Derived&>(*this).add(search, value, index);
        }
    }
    void update1_null(int64_t index=0) {
        null_count++;
        static_cast<Derived&>(*this).add_missing(index);
    }
    void update1_nan(int64_t index=0) {
        nan_count++;
        static_cast<Derived&>(*this).add_nan(index);
    }
    std::vector<value_type> keys() {
        std::vector<value_type> v;
        for(auto el : this->map) {
            value_type value = el.first;
            v.push_back(value);

        }
        return v;
    }
    std::map<value_type, int64_t> extract() {
        std::map<value_type, int64_t> m;
        for(auto el : this->map) {
            value_type value = el.first;
            m[value] = el.second;

        }
        return m;

    }
    Hashmap<value_type, int64_t> map;
    int64_t count;
    int64_t nan_count;
    int64_t null_count;
};

template<class T, template<typename, typename> typename Hashmap>
class counter : public hash_base<counter<T, Hashmap>, T, Hashmap> {
public:
    using typename hash_base<counter<T, Hashmap>, T, Hashmap>::value_type;

    void add_missing(int64_t index) {
    }
    void add_nan(int64_t index) {
    }
    void add(value_type& value, int64_t index) {
        this->map.emplace(value, 1);
    }
    template<class Bucket>
    void add(Bucket& bucket, value_type& value, int64_t index) {
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

template<class T, template<typename, typename> typename  Hashmap>
class ordered_set : public hash_base<ordered_set<T, Hashmap>, T, Hashmap> {
public:
    using typename hash_base<ordered_set<T, Hashmap>, T, Hashmap>::value_type;

    static ordered_set* create(std::map<value_type, int64_t> dict, int64_t count, int64_t nan_count, int64_t null_count) {
        ordered_set* set = new ordered_set;
        for(auto el : dict) {
            value_type value = el.first;
            set->map.emplace(value, el.second);
        }
        set->count = count;
        set->nan_count = nan_count;
        set->null_count = null_count;
        return set;
    }
    py::object isin(py::array_t<value_type>& values) {
        int64_t size = values.size();
        py::array_t<bool> result(size);
        auto input = values.template unchecked<1>();
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        for(int64_t i = 0; i < size; i++) {
            const value_type& value = input(i);
            if(custom_isnan(value)) {
                output(i) = this->nan_count > 0;
            } else {
                auto search = this->map.find(value);
                auto end = this->map.end();
                if(search == end) {
                    output(i) = false;
                } else {
                    output(i) = true;
                }
            }
        }
        return result;
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
    void add_nan(int64_t index) {
    }
    void add_missing(int64_t index) {
    }
    void add(value_type& value, int64_t index) {
        this->map.emplace(value, this->count);
        this->count++;
    }
    template<class Bucket>
    void add(Bucket& position, value_type& value, int64_t index) {
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
            value_type value = el.first;
            v[el.second] = value;

        }
        return v;
    }
};

template<class T, template<typename, typename> typename Hashmap>
class index_hash : public hash_base<index_hash<T, Hashmap>, T, Hashmap> {
public:
    using typename hash_base<index_hash<T, Hashmap>, T, Hashmap>::value_type;
    typedef hashmap<value_type, std::vector<int64_t>> MultiMap;

    py::array_t<int64_t> map_index(py::array_t<value_type, py::array::c_style>& values) {
        int64_t size = values.size();
        py::array_t<int64_t, py::array::c_style> result(size);
        map_index_write(values, result);
        return result;
    }
    template<typename result_type>
    bool map_index_write(py::array_t<value_type, py::array::c_style>& values, py::array_t<result_type, py::array::c_style>& output_array) {
        int64_t size = values.size();
        auto input = values.template unchecked<1>();
        auto output = output_array.template mutable_unchecked<1>();
        bool encountered_unknown = false;
        py::gil_scoped_release gil;
        for(int64_t i = 0; i < size; i++) {
            const value_type& value = input(i);
            if(custom_isnan(value)) {
                output(i) = nan_index;
                assert(this->nan_count > 0);
            } else {
                auto search = this->map.find(value);
                auto end = this->map.end();
                if(search == end) {
                    output(i) = -1;
                    encountered_unknown = true;
                } else {
                    output(i) = search->second;
                }
            }
        }
        return encountered_unknown;
    }
    py::array_t<int64_t> map_index_with_mask(py::array_t<value_type, py::array::c_style>& values, py::array_t<uint8_t, py::array::c_style>& mask) {
        int64_t size = values.size();
        py::array_t<int64_t, py::array::c_style> result(size);
        map_index_with_mask_write(values, mask, result);
        return result;
    }
    template<typename result_type>
    bool map_index_with_mask_write(py::array_t<value_type, py::array::c_style>& values, py::array_t<uint8_t, py::array::c_style>& mask, py::array_t<result_type, py::array::c_style>& output_array) {
        int64_t size = values.size();
        assert(values.size() == mask.size());
        auto input = values.template unchecked<1>();
        auto input_mask = mask.template unchecked<1>();
        auto output = output_array.template mutable_unchecked<1>();
        bool encountered_unknown = false;
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
                auto search = this->map.find(value);
                auto end = this->map.end();
                if(search == end) {
                    output(i) = -1;
                    encountered_unknown = true;
                } else {
                    output(i) = search->second;
                }
            }
        }
        return encountered_unknown;
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
    void add(value_type& value, int64_t index) {
        this->map.emplace(value, index);
        this->count++;
    }
    template<class Bucket>
    void add(Bucket& position, value_type& value, int64_t index) {
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

            value_type& value = el.first;
            // const value_type& value = elem.first;
            auto search = this->map.find(value);
            auto end = this->map.end();
            if(search == end) {
                // we have a duplicate that is not in the current map, so we insert the first element
                this->map.emplace(value, source[0]);
                if(source.size() > 1) {
                    std::vector<int64_t>& target = this->multimap[value];
                    target.insert(target.end(), source.begin()+1, source.end());
                }
            } else {
                // easy case, just merge the vectors
                std::vector<int64_t>& target = this->multimap[value];
                target.insert(target.end(), source.begin(), source.end());
            }
            this->count += source.size();
        }
        has_duplicates = has_duplicates || other.has_duplicates;
    }
    std::vector<value_type> keys() {
        std::vector<value_type> v(this->map.size());
        for(auto el : this->map) {
            value_type value = el.first;
            v[el.second] = value;

        }
        return v;
    }
    int64_t missing_index;
    int64_t nan_index;
    MultiMap multimap; // this stores only the duplicates
    bool has_duplicates;
};
} // namespace vaex

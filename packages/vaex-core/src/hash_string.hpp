#include "hash.hpp"
#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include "superstring.hpp"

namespace py = pybind11;

namespace vaex {
    // class hash_string {
    //     virtual void update(StringSequence* strings, int64_t start_index=0) = 0;
    //     virtual std::vector<value_type> keys();
    // };

template<class Derived, class T, class A=T, class V=T>
class hash_base {
public:
    using value_type = T;
    using storage_type = A;
    using storage_type_view = V;
    hash_base() : count(0), nan_count(0), null_count(0) {}  ;
    void update(StringSequence* strings, int64_t start_index=0) {
        py::gil_scoped_release gil;
        int64_t size = strings->length;
        for(int64_t i = 0; i < size; i++) {
                if(strings->is_null(i)) {
                    null_count++;
                    static_cast<Derived&>(*this).add_missing(start_index + i);
                } else {
                    auto value = strings->get(i);
                    this->update1(value, start_index + i);
                }
        }
    }
    void update1(std::string& value, int64_t index=0) {
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
    /*void update_with_mask(StringSequence* strings, py::array_t<bool>& masks, int64_t start_index=0) {
        py::gil_scoped_release gil;
        int64_t size = strings->length;
        auto m = masks.template unchecked<1>();
        assert(m.size() == size);
        for(int64_t i = 0; i < size; i++) {
                if(strings->is_null(i) || m[i]) {
                    null_count++;
                    static_cast<Derived&>(*this).add_missing(start_index + i);
                } else {
                    auto value = strings->get(i);
                    auto search = this->map.find(value);
                    auto end = this->map.end();
                    if(search == end) {
                        static_cast<Derived&>(*this).add(value, start_index + i);
                    } else {
                        static_cast<Derived&>(*this).add(search, value, start_index + i);
                    }
                }
        }
    }*/
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

    void add(storage_type_view& storage_view_value, int64_t index) {
        this->map.emplace(storage_view_value, 1);
    }

    void add_missing(int64_t index) {
    }

    template<class Bucket>
    void add(Bucket& bucket, storage_type_view& storage_view_value, int64_t index) {
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
    py::object isin(StringSequence* strings) {
        int64_t size = strings->length;
        py::array_t<bool> result(size);
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        if(strings->has_null()) {
            for(int64_t i = 0; i < size; i++) {
                if(strings->is_null(i)) {
                    output(i) = this->null_count > 0;
                } else {
                    const storage_type_view& value = strings->get(i);
                    auto search = this->map.find(value);
                    auto end = this->map.end();
                    if(search == end) {
                        output(i) = false;
                    } else {
                        output(i) = true;
                    }
                }
            }
        } else {
            for(int64_t i = 0; i < size; i++) {
                const storage_type_view& value = strings->get(i);
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

    void add_missing(int64_t index) {
    }

    void add(storage_type_view& storage_value, int64_t index) {
        this->map.emplace(storage_value, this->count++);
    }

    template<class Bucket>
    void add(Bucket& position, storage_type_view& storage_view_value, int64_t index) {
        // duplicates can be detected by getting the __len__
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
                // duplicates can be detected by getting the __len__
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

template<class T=string, class V=string>
class index_hash : public hash_base<index_hash<T>, T, T, V> {
public:
    using typename hash_base<index_hash<T>, T, T, V>::value_type;
    using typename hash_base<index_hash<T>, T, T, V>::storage_type;
    using typename hash_base<index_hash<T>, T, T, V>::storage_type_view;
    typedef hashmap<storage_type, std::vector<int64_t>> MultiMap;

    py::array_t<int64_t> map_index(StringSequence* strings) {
        int64_t size = strings->length;
        py::array_t<int64_t> result(size);
        auto output = result.template mutable_unchecked<1>();
        map_index_write(strings, result);
        return result;
    }
    template<typename result_type>
    bool map_index_write(StringSequence* strings, py::array_t<result_type>& output_array) {
        int64_t size = strings->length;
        auto output = output_array.template mutable_unchecked<1>();
        bool encountered_unknown = false;
        py::gil_scoped_release gil;
        // null and nan map to 0 and 1, and move the index up
        int64_t offset = 0; //(this->null_count > 0 ? 1 : 0);
        if(strings->has_null()) {
            for(int64_t i = 0; i < size; i++) {
                if(strings->is_null(i)) {
                    output(i) = missing_index;
                    assert(this->null_count > 0);
                } else {
                    const storage_type_view& value = strings->get(i);
                    auto search = this->map.find(value);
                    auto end = this->map.end();
                    if(search == end) {
                        output(i) = -1;
                        encountered_unknown = true;
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
                    encountered_unknown = true;
                } else {
                    output(i) = search->second + offset;
                }
            }
        }
        return encountered_unknown;
    }

    std::tuple<py::array_t<int64_t>, py::array_t<int64_t>> map_index_duplicates(StringSequence* strings, int64_t start_index) {
        std::vector<typename MultiMap::value_type> found; // should this be a reference to the value_type?
        std::vector<int64_t> indices;
        // found.reserve(strings->length);

        const auto end = this->multimap.end(); // we don't modify the multimap, so keep this const
        int64_t size = 0;
        {
            py::gil_scoped_release gil;
            if(strings->has_null()) {
                for(size_t i = 0; i < strings->length; i++) {
                    if(strings->is_null(i)) {
                    } else {
                        const storage_type_view& value = strings->get(i);
                        auto search = this->multimap.find(value);
                        if(search != end) {
                            found.push_back(*search);
                            size += search->second.size();
                            indices.insert(indices.end(), search->second.size(), start_index+i);
                        }
                    }
                }
            } else {
                for(size_t i = 0; i < strings->length; i++) {
                    const storage_type_view& value = strings->get(i);
                    auto search = this->multimap.find(value);
                    if(search != end) {
                        found.push_back(*search);
                        size += search->second.size();
                        indices.insert(indices.end(), search->second.size(), start_index+i);
                    }
                }
            }
        }

        py::array_t<int64_t> result(size);
        py::array_t<int64_t> indices_array(size);
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

    void add_missing(int64_t index) {
        this->missing_index = index;
    }

    void add(storage_type_view& storage_value, int64_t index) {
        this->map.emplace(storage_value, index);
        this->count++;
    }

    template<class Bucket>
    void add(Bucket& position, storage_type_view& storage_view_value, int64_t index) {
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
    int64_t missing_index;
    MultiMap multimap; // this stores only the duplicates
    bool has_duplicates;

};



}
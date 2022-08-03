#include "hash.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "superstring.hpp"
#include <Python.h>
#include <iostream>
#include <numpy/arrayobject.h>

namespace py = pybind11;
const int64_t chunk_size = 1024 * 128;

namespace vaex {
// class hash_string {
//     virtual void update(StringSequence* strings, int64_t start_index=0) = 0;
//     virtual std::vector<value_type> keys();
// };

template <class Derived, class T, class A = T, class V = T>
class hash_base : public hash_common<Derived, T, hashmap<T, int64_t>> {
  public:
    // using value_type = T;
    using Base = hash_common<Derived, T, hashmap<T, int64_t>>;
    using hashmap_type = typename Base::hashmap_type;
    using key_type = typename Base::key_type;
    using key_view_type = V;
    using value_type = typename Base::value_type;

    using storage_type = A;
    using storage_type_view = V;
    using key_type_view = V;
    // using hasher = typename Base::hasher;
    // using hasher_map = typename Base::hasher_map;
    using hasher_map_choice = typename Base::hasher_map_choice;

    hash_base(int nmaps = 1, int64_t limit = -1) : Base(nmaps, limit), string_arrays(0) {
        for (int i = 0; i < nmaps; i++) {
            string_arrays.emplace_back(std::make_shared<StringList64>());
            StringList64 *strings = string_arrays[i].get();
            // equal_to<string_ref>& eq = this->maps[i].key_eq();
            this->maps[i].m_ht.strings_equals = strings;
            this->maps[i].m_ht.strings_hash = strings;
        }
    };

    virtual std::string _get(hashmap_type &map, typename hashmap_type::key_type key) override { return map.m_ht.strings_equals->get(key.index); };

    size_t bytes_used() const {
        int64_t buffer_size = 0; // collect buffer size
        size_t bytes = 0;
        for (auto map : this->maps) {
            size_t buckets = map.size();
            bytes += buckets * (buffer_size + sizeof(value_type));
        }
        return bytes;
    }

    py::object update(StringSequence *strings, int64_t start_index = 0, int64_t chunk_size = 1024 * 16, int64_t bucket_size = 1024 * 128, bool return_values = false) {
        if (this->sealed) {
            throw std::runtime_error("cannot add to sealed hashmap");
        }
        if (bucket_size < chunk_size) {
            throw std::runtime_error("bucket size should be larger than chunk_size");
        }
        if (this->limit >= 0 && return_values) {
            throw std::runtime_error("Cannot combine limit with return_inverse");
        }
        int64_t size = strings->length;

        // for strings we always use offsets, we don't copy the data
        // const bool use_offsets = return_values;

        py::array_t<value_type> values_array(return_values ? size : 1);
        auto values_ptr = values_array.mutable_data(0);

        py::array_t<int16_t> values_map_index_array(return_values ? size : 1);
        auto values_map_index_ptr = values_map_index_array.mutable_data(0);

        {
            using offset_type = int32_t;

            bool full = false; // if we reached the limit, we can quickly get out

            py::gil_scoped_release gil;
            size_t nmaps = this->maps.size();
            int chunks = (size + chunk_size - 1) / chunk_size;
            std::vector<size_t> hashes(chunk_size);
            std::vector<std::vector<offset_type>> offsets(nmaps);
            std::vector<offset_type> null_offsets;

            {
                for (auto &offset : offsets) {
                    offset.reserve(chunk_size / nmaps);
                }
            }

            auto flush_bucket = [&](int16_t bucket_index) {
                auto &map = this->maps[bucket_index];
                int64_t i = 0;
                for (offset_type offset : offsets[bucket_index]) {
                    // key_type key = keys[offset];
                    string_view key = strings->view(offset);
                    auto search = map.find(key);
                    auto end = map.end();
                    auto key_offset = offsets[bucket_index][i];
                    value_type value;
                    if (search == end) {
                        value = static_cast<Derived &>(*this).add_new(bucket_index, key, key_offset + start_index);
                    } else {
                        value = static_cast<Derived &>(*this).add_existing(search, bucket_index, key, key_offset + start_index);
                    }
                    if (return_values) {
                        values_ptr[key_offset] = value;
                        values_map_index_ptr[key_offset] = bucket_index;
                    }
                    i++;
                }
                offsets[bucket_index].clear();
            };

            for (int64_t chunk = 0; chunk < chunks; chunk++) {
                int64_t i1 = chunk * chunk_size;
                int64_t i2 = std::min(size, (chunk + 1) * chunk_size);

                for (int64_t i = i1; i < i2; i++) {
                    auto value = strings->view(i);
                    hashes[i - i1] = hasher_map_choice()(value);
                }
                for (int64_t i = i1; i < i2; i++) {
                    if (strings->is_null(i)) {
                        null_offsets.push_back(i);
                    } else {
                        // key_type_view value = strings->view(i);
                        std::size_t hash = hashes[i - i1];
                        // std::size_t hash = hasher_map_choice()(value);
                        size_t map_index = (hash % nmaps);
                        offsets[map_index].push_back(i);
                    }
                }
            }
            if (this->limit >= 0) {
                if (this->count() >= this->limit) {
                    full = true;
                }
            }
            if (!full) {
                for (size_t map_index = 0; map_index < nmaps; map_index++) {
                    if (offsets[map_index].size()) {
                        const std::lock_guard<std::mutex> lock(this->maplocks[map_index]);
                        flush_bucket(map_index);
                    }
                }
            }
            if (null_offsets.size()) {
                // null and nan go into map 0 (as if the hash is always 0)
                const std::lock_guard<std::mutex> lock(this->maplocks[0]);
                if (return_values) {
                    for (int32_t offset : null_offsets) {
                        value_type value = this->update1_null(offset);
                        if (return_values) {
                            values_ptr[offset] = value;
                            values_map_index_ptr[offset] = 0;
                        }
                    }
                } else {
                    for (int32_t offset : null_offsets) {
                        this->update1_null(offset);
                    }
                }
            }
        } // gil release
        if (return_values) {
            return py::make_tuple(values_array, values_map_index_array);
        } else {
            return py::none();
        }
    }
    std::shared_ptr<StringList64> key_array() {
        int nmaps = this->maps.size();
        if (nmaps == 1) {
            return string_arrays[0];
        } else {
            // std::vector<int64_t> lengths(size);
            size_t total_length = 0;
            size_t byte_length = 0;
            for (int i = 0; i < nmaps; i++) {
                total_length += string_arrays[i]->length;
                byte_length += string_arrays[i]->indices[string_arrays[i]->length];
            }
            std::shared_ptr<StringList64> result = std::make_shared<StringList64>(byte_length, total_length);
            StringList64 *sl = result.get();
            sl->indices[0] = 0;
            int64_t indices_offset = 0;
            int64_t byte_offset = 0;
            for (int i = 0; i < nmaps; i++) {
                int64_t byte_size = string_arrays[i]->indices[string_arrays[i]->length];
                std::copy(string_arrays[i]->bytes, string_arrays[i]->bytes + byte_size, sl->bytes + byte_offset);
                int64_t length = string_arrays[i]->length;
                std::transform(string_arrays[i]->indices + 1, string_arrays[i]->indices + 1 + length, sl->indices + indices_offset + 1, [&](int64_t i) { return i + byte_offset; });
                byte_offset += byte_size;
                indices_offset += length;
            }
            if (this->null_count) {
                sl->ensure_null_bitmap();
                sl->set_null(this->null_index());
            }
            return result;
        }
    }

    std::vector<std::map<std::string, value_type>> extract() {
        std::vector<std::map<std::string, value_type>> map_vector;
        int i = 0;
        for (auto &map : this->maps) {
            std::map<std::string, value_type> m;
            auto strings = string_arrays[i++];
            for (auto &el : map) {
                string_view view = strings->view(el.first.index);
                std::string value(view);
                m[value] = el.second;
            }
            map_vector.push_back(std::move(m));
        }
        return map_vector;
    }
    std::vector<std::shared_ptr<StringList64>> string_arrays;
};

template <class T = string_ref, class A = T, class V = string_ref>
class counter : public hash_base<counter<T, A>, T, A, V>, public counter_mixin<T, A, counter<T, A>> {
  public:
    using Base = hash_base<counter<T, A>, T, A, V>;
    using typename Base::hashmap_type;
    using typename Base::key_type;
    using typename Base::key_type_view;
    using typename Base::storage_type;
    using typename Base::storage_type_view;
    using typename Base::value_type;

    counter(int nmaps = 1) : Base(nmaps), null_value(0x7fffffff) {}

    template <class Bucket>
    int64_t key_offset(int64_t natural_order, int16_t map_index, Bucket &bucket, int64_t map_offset) {
        return natural_order + (this->null_count > 0 ? 1 : 0);
    }
    virtual value_type nan_index() const { return -1; }
    virtual value_type null_index() const { return null_value; }

    value_type add_null(int64_t index) {
        // we only add it the first time
        if (this->null_count == 1) {
            null_value = this->maps[0].size();
            auto &string_array = this->string_arrays[0];
            string_array->push_null();
        }
        // parent already counts this
        return this->null_count;
    }
    value_type add_nan(int64_t index) {
        // same
        return this->null_count;
    }
    int64_t value_null() { return this->null_count; }
    int64_t value_nan() { return this->nan_count; }
    value_type add_new(int16_t map_index, string_view key, int64_t index, int64_t value=1) {
        auto &map = this->maps[map_index];
        auto &string_array = this->string_arrays[map_index];
        string_array->push(key);
        string_ref persistent_value(string_array->length - 1);
        map.emplace(persistent_value, value);
        return 1;
    }
    template <class Bucket>
    value_type add_existing(Bucket &bucket, int16_t map_index, string_view key, int64_t index) {
        set_second(bucket, bucket->second + 1);
        return bucket->second;
    }
    py::object counts() {
        py::array_t<value_type> output_array(this->length());
        auto output = output_array.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        auto offsets = this->offsets();
        size_t map_index = 0;
        int64_t natural_order = 0;
        // TODO: can be parallel due to non-overlapping maps
        for (auto &map : this->maps) {
            for (auto &el : map) {
                key_type key = el.first;
                value_type value = el.second;
                output(key.index) = value;
            }
            // map_index += 1;
        }
        // no nans possible
        // if (this->nan_count) {
        //     output(this->nan_index()) = this->nan_count;
        // }
        if (this->null_count) {
            output(this->null_index()) = this->null_count;
        }
        return output_array;
    }
    void merge(const counter &other) {
        py::gil_scoped_release gil;
        if (this->maps.size() != other.maps.size()) {
            throw std::runtime_error("cannot merge with an unequal maps");
        }

        for (size_t i = 0; i < this->maps.size(); i++) {
            auto &strings = other.string_arrays[i];
            for (auto &elem : other.maps[i]) {
                const key_type &ref = elem.first;
                string_view key = strings->view(ref.index);
                auto search = this->maps[i].find(key);
                auto end = this->maps[i].end();
                if (search == end) {
                    this->add_new(i, key, 0, elem.second);
                } else {
                    set_second(search, search->second + elem.second);
                }
            }
        }
        if (other.null_count) {
            this->update1_null();
            this->null_count += other.null_count - 1;
        } else {
            this->null_count += other.null_count;
        }
        // no nans
        // this->nan_count += other.nan_count;
    }
    value_type null_value;
};

template <class T = string_ref, class V = string_ref>
class ordered_set : public hash_base<ordered_set<T>, T, T, V> {
  public:
    using Base = hash_base<ordered_set<T>, T, T, V>;
    using typename Base::hasher_map_choice;
    using typename Base::hashmap_type;
    using typename Base::key_type;
    using typename Base::key_type_view;
    using typename Base::storage_type;
    using typename Base::storage_type_view;
    using typename Base::value_type;

    ordered_set(int nmaps = 1, int64_t limit = -1) : Base(nmaps, limit), null_value(0x7fffffff), ordinal_code_offset_null(0) {}

    virtual value_type nan_index() const { return -1; }
    virtual value_type null_index() const { return null_value; }

    template <class Bucket>
    int64_t key_offset(int64_t natural_order, int16_t map_index, Bucket &bucket, int64_t offset) {
        int64_t index = bucket.second + offset;
        return index;
    }
    value_type add_null(int64_t index) {
        // we only add it the first time
        if (this->null_count == 1) {
            null_value = this->maps[0].size();
            auto &string_array = this->string_arrays[0];
            string_array->push_null();
            ordinal_code_offset_null++;
        }
        return null_value;
    }

    value_type add_new(int16_t map_index, string_view storage_value, int64_t index) {
        auto &map = this->maps[map_index];
        value_type ordinal_code = map.size();
        if (map_index == 0) {
            ordinal_code += ordinal_code_offset_null;
        }
        auto &string_array = this->string_arrays[map_index];
        string_array->push(storage_value);
        // storage_type_view persistent_value = string_array->end();
        string_ref persistent_value(string_array->length - 1);
        map.emplace(persistent_value, ordinal_code);
        return ordinal_code;
    }

    template <class Bucket>
    value_type add_existing(Bucket &bucket, int16_t map_index, string_view storage_view_value, int64_t index) {
        return bucket->second;
    }

    template <class SL>
    static ordered_set *create(std::shared_ptr<SL> keys, int64_t null_value, int64_t nan_count, int64_t null_count, std::string *fingerprint) {
        ordered_set *set = new ordered_set(1);
        set->maps[0].m_ht.strings_equals = keys.get();
        set->maps[0].m_ht.strings_hash = keys.get();
        set->string_arrays[0] = keys;
        {
            size_t size = keys->length;
            set->maps[0].reserve(size);
            py::gil_scoped_release gil;
            for (size_t i = 0; i < size; i++) {
                if (keys->is_null(i)) {
                    set->null_count++;
                    set->null_value = i;
                    set->ordinal_code_offset_null = 1;
                    set->update1_null();
                } else {
                    string_ref ref(i);
                    set->maps[0].emplace(ref, i);
                }
            }
        }
        if (set->count() != (int64_t)keys->length) {
            throw std::runtime_error(std::string("key array of length ") + std::to_string(keys->length) + " does not match expected length of " + std::to_string(set->count()));
        }
        if (nan_count == 0) {
            if (set->nan_count != 0) {
                throw std::runtime_error("NaN found in data, while claiming there should be none");
            }
        } else {
            if (set->nan_count == 0) {
                throw std::runtime_error("no NaN found in data, while claiming there should be");
            }
        }
        if (null_count == 0) {
            if (set->null_count != 0) {
                throw std::runtime_error("null found in data, while claiming there should be none");
            }
        } else {
            if (set->null_count == 0) {
                throw std::runtime_error("no null found in data, while claiming there should be");
            }
            if (set->null_value != null_value) {
                throw std::runtime_error(std::string("null_value = ") + std::to_string(set->null_value) + " does not match expected value = " + std::to_string(null_value));
            }
        }
        set->null_count = null_count;
        set->nan_count = nan_count;
        set->sealed = true;
        if (fingerprint) {
            set->fingerprint = *fingerprint;
        }
        return set;
    }
    // virtual value_type
    py::object isin(StringSequence *strings) {
        int64_t size = strings->length;
        py::array_t<bool> result(size);
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        size_t nmaps = this->maps.size();
        if (strings->has_null()) {
            for (int64_t i = 0; i < size; i++) {
                if (strings->is_null(i)) {
                    output(i) = this->null_count > 0;
                } else {
                    const string_view &value = strings->view(i);
                    std::size_t hash = hasher_map_choice()(value);
                    size_t map_index = (hash % nmaps);
                    auto search = this->maps[map_index].find(value);
                    auto end = this->maps[map_index].end();
                    if (search == end) {
                        output(i) = false;
                    } else {
                        output(i) = true;
                    }
                }
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                const string_view &value = strings->view(i);
                std::size_t hash = hasher_map_choice()(value);
                size_t map_index = (hash % nmaps);
                auto search = this->maps[map_index].find(value);
                auto end = this->maps[map_index].end();
                if (search == end) {
                    output(i) = false;
                } else {
                    output(i) = true;
                }
            }
        }
        return result;
    }

    virtual void map_many(StringSequence *strings, int64_t offset, int64_t length, int64_t *output) override {
        // int64_t size = strings->length;
        size_t nmaps = this->maps.size();
        auto offsets = this->offsets();

        // split slow and fast path
        if (strings->has_null()) {
            for (int64_t i = offset; i < offset + length; i++) {
                if (strings->is_null(i)) {
                    output[i - offset] = this->null_value;
                    assert(this->null_count > 0);
                } else {
                    const string_view &key = strings->view(i);
                    size_t hash = hasher_map_choice()(key);
                    size_t map_index = (hash % nmaps);
                    auto search = this->maps[map_index].find(key, hash);
                    auto end = this->maps[map_index].end();
                    if (search == end) {
                        output[i - offset] = -1;
                    } else {
                        output[i - offset] = search->second + offsets[map_index];
                    }
                }
            }

        } else {
            for (int64_t i = offset; i < offset + length; i++) {
                const string_view &key = strings->view(i);
                std::size_t hash = hasher_map_choice()(key);
                size_t map_index = (hash % nmaps);
                auto search = this->maps[map_index].find(key, hash);
                auto end = this->maps[map_index].end();
                if (search == end) {
                    output[i - offset] = -1;
                } else {
                    output[i - offset] = search->second + offsets[map_index];
                }
            }
        }
    };
    virtual int64_t map_key(string_ref key) {
        size_t nmaps = this->maps.size();
        auto offsets = this->offsets();
        size_t hash = hasher_map_choice()(key);
        size_t map_index = (hash % nmaps);
        auto search = this->maps[map_index].find(key);
        auto end = this->maps[map_index].end();
        if (search == end) {
            return -1;
        } else {
            return search->second + offsets[map_index];
        }
    }

    py::object map_ordinal(StringSequence *strings) {
        size_t size = this->length();
        if (size < (1u << 7u)) {
            return this->template _map_ordinal<int8_t>(strings);
        } else if (size < (1u << 15u)) {
            return this->template _map_ordinal<int16_t>(strings);
        } else if (size < (1u << 31u)) {
            return this->template _map_ordinal<int32_t>(strings);
        } else {
            return this->template _map_ordinal<int64_t>(strings);
        }
    }
    template <class OutputType>
    py::array_t<OutputType> _map_ordinal(StringSequence *strings) {
        int64_t size = strings->length;
        py::array_t<OutputType> result(size);
        if (size == 0) {
            return result;
        }
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        size_t nmaps = this->maps.size();
        auto offsets = this->offsets();

        if (nmaps == 1) {
            auto &map0 = this->maps[0];
            // split slow and fast path
            if (strings->has_null()) {
                for (int64_t i = 0; i < size; i++) {
                    if (strings->is_null(i)) {
                        output(i) = this->null_value;
                        assert(this->null_count > 0);
                    } else {
                        const string_view &key = strings->view(i);
                        auto search = map0.find(key);
                        auto end = map0.end();
                        if (search == end) {
                            output(i) = -1;
                        } else {
                            output(i) = search->second;
                        }
                    }
                }
            } else {
                for (int64_t i = 0; i < size; i++) {
                    const string_view &key = strings->view(i);
                    auto search = map0.find(key);
                    auto end = map0.end();
                    if (search == end) {
                        output(i) = -1;
                    } else {
                        output(i) = search->second;
                    }
                }
            }
        } else {
            // split slow and fast path
            if (strings->has_null()) {
                for (int64_t i = 0; i < size; i++) {
                    if (strings->is_null(i)) {
                        output(i) = this->null_value;
                        assert(this->null_count > 0);
                    } else {
                        const string_view &key = strings->view(i);
                        size_t hash = hasher_map_choice()(key);
                        size_t map_index = (hash % nmaps);
                        auto search = this->maps[map_index].find(key, hash);
                        auto end = this->maps[map_index].end();
                        if (search == end) {
                            output(i) = -1;
                        } else {
                            output(i) = search->second + offsets[map_index];
                        }
                    }
                }
            } else {
                for (int64_t i = 0; i < size; i++) {
                    const string_view &key = strings->view(i);
                    std::size_t hash = hasher_map_choice()(key);
                    size_t map_index = (hash % nmaps);
                    auto search = this->maps[map_index].find(key, hash);
                    auto end = this->maps[map_index].end();
                    if (search == end) {
                        output(i) = -1;
                    } else {
                        output(i) = search->second + offsets[map_index];
                    }
                }
            }
        }
        return result;
    }

    void merge(std::vector<ordered_set *> others) {
        if (this->sealed) {
            throw std::runtime_error("hashmap is sealed, cannot merge");
        }
        for (auto &other : others) {
            if (this->maps.size() != other->maps.size()) {
                throw std::runtime_error("cannot merge with an unequal maps");
            }
        }
        py::gil_scoped_release gil;
        for (auto &other : others) {
            if (other->null_count) {
                // TODO: ugly, see add_null
                auto save = this->null_count;
                this->null_count = 1;
                this->add_null(this->maps[0].size());
                this->null_count = save;
                this->null_count += other->null_count;
            }
            for (size_t i = 0; i < this->maps.size(); i++) {
                auto strings = other->string_arrays[i];
                for (auto &elem : other->maps[i]) {
                    const key_type &ref = elem.first;
                    string_view key = strings->view(ref.index);
                    auto search = this->maps[i].find(key);
                    auto end = this->maps[i].end();
                    if (search == end) {
                        this->add_new(i, key, this->maps[i].size());
                    } else {
                    }
                }
                other->maps[i].clear();
            }
        }
    }

    value_type null_value;
    value_type ordinal_code_offset_null;
};

template <class T = string_ref, class V = string_ref>
class index_hash : public hash_base<index_hash<T>, T, T, V> {
  public:
    using Base = hash_base<index_hash<T>, T, T, V>;
    using typename Base::hasher_map_choice;
    using typename Base::hashmap_type;
    using typename Base::key_type;
    using typename Base::storage_type;
    using typename Base::storage_type_view;
    using typename Base::value_type;
    typedef hashmap<key_type, std::vector<int64_t>> overflow_type;

    // TODO: might be better to use a node based hasmap, we don't want to move large vectors
    index_hash(int nmaps, int64_t limit = -1) : Base(nmaps, limit), overflows(nmaps), has_duplicates(false), null_value(-1) {
        for (int i = 0; i < nmaps; i++) {
            // string_arrays_overflow.emplace_back(std::make_shared<StringList64>());
            // for each key in overflow, it should be present in the main string array
            StringList64 *strings = this->string_arrays[i].get();
            // equal_to<string_ref>& eq = this->maps[i].key_eq();
            overflows[i].m_ht.strings_equals = strings;
            overflows[i].m_ht.strings_hash = strings;
        }
    }

    int64_t length() const {
        int64_t c = this->count();
        for (const overflow_type &overflow : overflows) {
            // overflow is a hashmap, which maps a key to a vector
            for (auto &el : overflow) {
                c += el.second.size();
            }
        }
        return c;
    }

    template <class Bucket>
    int64_t key_offset(int64_t natural_order, int16_t map_index, Bucket &bucket, int64_t map_offset) {
        return natural_order + (this->null_count > 0 ? 1 : 0);
    }
    virtual value_type nan_index() const { return -1; }
    virtual value_type null_index() const { return 0; }

    value_type add_null(int64_t index) {
        this->null_value = index;
        return index;
    }

    value_type add_new(int16_t map_index, string_view key, int64_t index) {
        auto &map = this->maps[map_index];
        auto &string_array = this->string_arrays[map_index];
        string_array->push(key);
        string_ref persistent_value(string_array->length - 1);
        map.emplace(persistent_value, index);
        return index;
    }

    template <class Bucket>
    value_type add_existing(Bucket &position, int16_t map_index, string_view key, int64_t index) {
        // we found a duplicate, add it to overflow
        auto &overflow = overflows[map_index];
        overflow[position->first].push_back(index);
        has_duplicates = true;
        return index;
    }

    py::array_t<int64_t> map_index(StringSequence *strings) {
        int64_t size = strings->length;
        py::array_t<int64_t> result(size);
        map_index_write(strings, result);
        return result;
    }
    template <typename result_type>
    bool map_index_write(StringSequence *strings, py::array_t<result_type> &output_array) {
        int64_t size = strings->length;
        auto output = output_array.template mutable_unchecked<1>();
        bool encountered_unknown = false;
        py::gil_scoped_release gil;
        // null and nan map to 0 and 1, and move the index up
        int64_t offset = 0; //(this->null_count > 0 ? 1 : 0);
        size_t nmaps = this->maps.size();
        if (strings->has_null()) {
            for (int64_t i = 0; i < size; i++) {
                if (strings->is_null(i)) {
                    output(i) = null_value;
                    assert(this->null_count > 0);
                    if(null_value == -1) {
                        encountered_unknown = true;
                    }
                } else {
                    const string_view &key = strings->view(i);
                    std::size_t hash = hasher_map_choice()(key);
                    size_t map_index = (hash % nmaps);
                    auto search = this->maps[map_index].find(key);
                    auto end = this->maps[map_index].end();
                    if (search == end) {
                        output(i) = -1;
                        encountered_unknown = true;
                    } else {
                        output(i) = search->second + offset;
                    }
                }
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                const string_view &key = strings->view(i);
                std::size_t hash = hasher_map_choice()(key);
                size_t map_index = (hash % nmaps);
                auto search = this->maps[map_index].find(key);
                auto end = this->maps[map_index].end();
                if (search == end) {
                    output(i) = -1;
                    encountered_unknown = true;
                } else {
                    output(i) = search->second + offset;
                }
            }
        }
        return encountered_unknown;
    }

    std::tuple<py::array_t<int64_t>, py::array_t<int64_t>> map_index_duplicates(StringSequence *strings, int64_t start_index) {
        std::vector<typename overflow_type::value_type> found; // should this be a reference to the value_type?
        std::vector<int64_t> indices;

        size_t nmaps = this->maps.size();
        int64_t size = 0;
        {
            py::gil_scoped_release gil;
            if (strings->has_null()) {
                for (size_t i = 0; i < strings->length; i++) {
                    if (strings->is_null(i)) {
                    } else {
                        const string_view &key = strings->view(i);
                        std::size_t hash = hasher_map_choice()(key);
                        size_t map_index = (hash % nmaps);
                        auto search = this->overflows[map_index].find(key);
                        auto end = this->overflows[map_index].end();
                        if (search != end) {
                            found.push_back(*search);
                            size += search->second.size();
                            indices.insert(indices.end(), search->second.size(), start_index + i);
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < strings->length; i++) {
                    const string_view &key = strings->view(i);
                    std::size_t hash = hasher_map_choice()(key);
                    size_t map_index = (hash % nmaps);
                    auto search = this->overflows[map_index].find(key);
                    auto end = this->overflows[map_index].end();
                    if (search != end) {
                        found.push_back(*search);
                        size += search->second.size();
                        indices.insert(indices.end(), search->second.size(), start_index + i);
                    }
                }
            }
        }

        py::array_t<int64_t> result(size);
        py::array_t<int64_t> indices_array(size);
        auto output = result.template mutable_unchecked<1>();
        auto output_indices = indices_array.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        size_t index = 0;

        std::copy(indices.begin(), indices.end(), &output_indices(0));

        for (auto el : found) {
            std::vector<int64_t> &indices = el.second;
            for (int64_t i : indices) {
                output(index++) = i;
            }
        }
        return std::make_tuple(indices_array, result);
    }

    py::object extract() {
        py::dict m;
        int16_t map_index = 0;
        int i = 0;
        for (auto &map : this->maps) {
            auto strings = this->string_arrays[i++];
            for (auto &el : map) {
                key_type key = el.first;
                value_type value = el.second;
                string_view key_view = strings->view(el.first.index);
                std::string str(key_view);
                // if multiple found, we add a list
                if (overflows[map_index].count(key_view)) {
                    py::list l;
                    l.append(value);
                    for (value_type v : overflows[map_index].find(key_view)->second) {
                        l.append(v);
                    }
                    m[str.c_str()] = l;
                } else {
                    m[str.c_str()] = value;
                }
            }
            map_index++;
        }
        return std::move(m);
    }

    void merge(const index_hash &other) {
        py::gil_scoped_release gil;
        if (this->maps.size() != other.maps.size()) {
            throw std::runtime_error("cannot merge with an unequal maps");
        }

        // first, merge the primary maps
        for (size_t i = 0; i < this->maps.size(); i++) {
            for (auto &elem : other.maps[i]) {
                // we get a string_view, since we want to find the key in another
                // hashmap
                string_view key = other.string_arrays[i]->view(elem.first.index);
                auto search = this->maps[i].find(key);
                auto end = this->maps[i].end();
                // if not found, we simply add it to the normal map
                if (search == end) {
                    this->add_new(i, key, elem.second);
                } else {
                    // if already in, add it to the overflow
                    this->add_existing(search, i, key, elem.second);
                }
            }
        }
        this->nan_count += other.nan_count;
        this->null_count += other.null_count;
        // now we merge the overflow
        for (size_t i = 0; i < this->maps.size(); i++) {
            auto &string_array_other = other.string_arrays[i];
            for (auto el : other.overflows[i]) {
                std::vector<int64_t> &source = el.second;

                const key_type &key_ref = el.first;
                string_view key = string_array_other->view(key_ref.index);

                auto search = this->maps[i].find(key);
                auto end = this->maps[i].end();
                if (search == end) {
                    // we have a duplicate that is not in the current map, so we insert the first element
                    this->add_new(i, key, source[0]);
                    if (source.size() > 1) {
                        // the rest can go into overflow
                        const key_type key_ref_this(this->string_arrays[i]->length - 1);
                        std::vector<int64_t> &target = this->overflows[i][key_ref_this];
                        target.insert(target.end(), source.begin() + 1, source.end());
                    }
                } else {
                    // easy case, just merge the vectors
                    std::vector<int64_t> &target = this->overflows[i][search->first];
                    target.insert(target.end(), source.begin(), source.end());
                }
            }
        }
        has_duplicates = has_duplicates || other.has_duplicates;
    }

    std::vector<overflow_type> overflows; // this stores only the duplicates
    int64_t null_value;
    bool has_duplicates;
};

} // namespace vaex
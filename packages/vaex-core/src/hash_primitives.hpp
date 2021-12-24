#include "hash.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <limits>
#include <mutex>
#include <numpy/arrayobject.h>

namespace py = pybind11;
#define custom_isnan(value) (!(value == value))

namespace vaex {

template <class Key, class Value>
using hashmap_primitive = hashmap<Key, Value>;

template <class Key, class Value>
using hashmap_primitive_pg = hashmap_pg<Key, Value>;

template <typename T>
struct NaNish {
    static constexpr T value = -1;
};

template <>
struct NaNish<float> {
    static constexpr float value = std::numeric_limits<float>::quiet_NaN();
};

template <>
struct NaNish<double> {
    static constexpr double value = std::numeric_limits<double>::quiet_NaN();
};

template <>
struct NaNish<bool> {
    static constexpr bool value = false;
};

const int64_t bucket_size = 1024 * 64;

template <class Derived, class T, template <typename, typename> class Hashmap>
class hash_base : public hash_common<Derived, T, Hashmap<T, int64_t>> {
  public:
    using Base = hash_common<Derived, T, Hashmap<T, int64_t>>;
    using hashmap_type = typename Base::hashmap_type;
    using key_type = typename Base::key_type;
    using value_type = typename Base::value_type;
    using hasher = typename Base::hasher;
    using hasher_map = typename Base::hasher_map;
    using hasher_map_choice = typename Base::hasher_map_choice;

    hash_base(int nmaps = 1, int64_t limit = -1) : Base(nmaps, limit){};
    // void reserve(int64_t count_) {
    //     py::gil_scoped_release gil;
    //     for(auto map : this->maps) {
    //         map.reserve((count_ + maps.size() - 1) / maps.size());
    //     }
    // }
    size_t bytes_used() const {
        size_t bytes = 0;
        for (auto map : this->maps) {
            size_t buckets = map.size();
            bytes += buckets * (sizeof(key_type) + sizeof(value_type));
        }
        return bytes;
    }
    py::object update(py::array_t<key_type> &keys, int64_t start_index = 0, int64_t chunk_size = 1024 * 16, int64_t bucket_size = 1024 * 128, bool return_values = false) {
        size_t size = keys.size();
        const key_type *keys_ptr = keys.data(0);
        if (keys.strides()[0] != keys.itemsize()) {
            throw std::runtime_error("stride not equal to bytesize");
        }
        return _update(size, keys_ptr, nullptr, start_index = start_index, chunk_size = chunk_size, bucket_size = bucket_size, return_values = return_values);
    }

    py::object update_with_mask(py::array_t<key_type> &keys, py::array_t<bool> &masks, int64_t start_index = 0, int64_t chunk_size = 1024 * 16, int64_t bucket_size = 1024 * 128,
                                bool return_values = false) {
        if (keys.size() != masks.size()) {
            throw std::runtime_error("array and mask should be of same size");
        }
        size_t size = keys.size();
        const key_type *keys_ptr = keys.data(0);
        const bool *mask_ptr = masks.data(0);
        if (keys.strides()[0] != keys.itemsize()) {
            throw std::runtime_error("stride not equal to bytesize");
        }
        if (masks.strides()[0] != masks.itemsize()) {
            throw std::runtime_error("stride not equal to bytesize for mask");
        }
        return _update(size, keys_ptr, mask_ptr, start_index = start_index, chunk_size = chunk_size, bucket_size = bucket_size, return_values = return_values);
    }

    py::object _update(int64_t size, const key_type *keys, const bool *masks, int64_t start_index = 0, int64_t chunk_size = 1024 * 16, int64_t bucket_size = 1024 * 128, bool return_values = false) {
        if (bucket_size < chunk_size) {
            throw std::runtime_error("bucket size should be larger than chunk_size");
        }
        if (this->limit >= -1 && return_values) {
            throw std::runtime_error("Cannot combine limit with return_inverse");
        }
        const bool use_offsets = return_values || (start_index != -1);

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
            // int64_t bucket_size_try = chunk_size/2;
            // std::vector<size_t> hashes(chunk_size);
            std::vector<std::vector<key_type>> buckets(nmaps);
            std::vector<std::vector<offset_type>> offsets(nmaps);
            std::vector<offset_type> nan_offsets;
            std::vector<offset_type> null_offsets;
            {
                int64_t j = 0;
                for (auto &offset : offsets) {
                    buckets[j++].reserve(chunk_size / nmaps);
                    if (use_offsets) {
                        offset.reserve(chunk_size / nmaps);
                    }
                }
            }
#pragma GCC visibility push(hidden)
            auto flush_bucket = [&](int16_t bucket_index) {
                auto &map = this->maps[bucket_index];
                if (use_offsets) {
                    int64_t i = 0;
                    // for(offset_type offset : offsets[bucket_index]) {
                    //     key_type key = keys[offset];
                    for (key_type key : buckets[bucket_index]) {
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
                } else {
                    for (key_type key : buckets[bucket_index]) {
                        auto search = map.find(key);
                        auto end = map.end();
                        if (search == end) {
                            static_cast<Derived &>(*this).add_new(bucket_index, key, 0);
                        } else {
                            static_cast<Derived &>(*this).add_existing(search, bucket_index, key, 0);
                        }
                    }
                }
                buckets[bucket_index].clear();
                if (use_offsets) {
                    offsets[bucket_index].clear();
                }
            };

#pragma GCC visibility pop
            for (int64_t chunk = 0; chunk < chunks; chunk++) {
                int64_t i1 = chunk * chunk_size;
                int64_t i2 = std::min(size, (chunk + 1) * chunk_size);

                // benchmarking never revealed faster results (maybe depends on cache size)
                // for(size_t map_index = 0; map_index < nmaps; map_index++) {
                //     auto & bucket = buckets[map_index];
                //     // if we are at risk of overflowing the buckets (causing malloc/remalloc), we have to flush
                //     if(int64_t(bucket.size()) >= (bucket_size - (i2-i1))) {
                //         const std::lock_guard<std::mutex> lock(maplocks[map_index]);
                //         flush_bucket(map_index);
                //     }
                //     // if we are pretty full, we try a flush
                //     if(int64_t(bucket.size()) >= bucket_size_try ) {
                //         if(maplocks[map_index].try_lock()) {
                //             flush_bucket(map_index);
                //             maplocks[map_index].unlock();
                //         }
                //     }
                // }

                if (std::numeric_limits<key_type>::is_integer && masks == nullptr && !use_offsets) {
                    for (int64_t i = i1; i < i2; i++) {
                        key_type value = keys[i];
                        std::size_t hash = hasher_map_choice()(value);
                        size_t map_index = (hash % nmaps);
                        buckets[map_index].push_back(value);
                    }
                } else if (std::numeric_limits<key_type>::is_integer && masks == nullptr && use_offsets) {
                    for (int64_t i = i1; i < i2; i++) {
                        key_type value = keys[i];
                        std::size_t hash = hasher_map_choice()(value);
                        size_t map_index = (hash % nmaps);
                        buckets[map_index].push_back(value);
                        offsets[map_index].push_back(i);
                    }
                } else {
                    for (int64_t i = i1; i < i2; i++) {
                        key_type value = keys[i];
                        if (masks && masks[i]) {
                            null_offsets.push_back(i);
                        } else if (custom_isnan(value)) {
                            nan_offsets.push_back(i);
                        } else {
                            // std::size_t hash = hashes[i-i1];
                            std::size_t hash = hasher_map_choice()(value);
                            size_t map_index = (hash % nmaps);
                            buckets[map_index].push_back(value);
                            if (use_offsets) {
                                offsets[map_index].push_back(i);
                            }
                        }
                    }
                }
            }
            // write out bucket to each corresponding map
            // this was an interesting idea, didn't notice speedups
            // std::vector<size_t> map_indices(nmaps);
            // std::iota(map_indices.begin(), map_indices.end(), 0);
            // while(!map_indices.empty()) {
            //     {
            //         int64_t index = std::rand() % map_indices.size();
            //         size_t map_index = map_indices[index];
            //         // if(maplocks[map_index].try_lock()) {
            //         {
            //             const std::lock_guard<std::mutex> lock(maplocks[map_index]);
            //             map_indices.erase(map_indices.begin() + index);
            //             flush_bucket(map_index);
            //             // maplocks[map_index].unlock();
            //         }
            //     }
            // }
            if(this->limit >= 0) {
                if(this->count() >= this->limit) {
                    full = true;
                }
            }
            if(!full) {
                for (size_t map_index = 0; map_index < nmaps; map_index++) {
                    if (buckets[map_index].size()) {
                        const std::lock_guard<std::mutex> lock(this->maplocks[map_index]);
                        flush_bucket(map_index);
                    }
                }
            }
            if (null_offsets.size() || nan_offsets.size()) {
                // null and nan go into map 0 (as if the hash is always 0)
                const std::lock_guard<std::mutex> lock(this->maplocks[0]);
                if (use_offsets) {
                    for (int32_t offset : null_offsets) {
                        value_type value = this->update1_null(offset);
                        if (return_values) {
                            values_ptr[offset] = value;
                            values_map_index_ptr[offset] = 0;
                        }
                    }
                    for (int32_t offset : nan_offsets) {
                        value_type value = this->update1_nan(offset);
                        if (return_values) {
                            values_ptr[offset] = value;
                            values_map_index_ptr[offset] = 0;
                        }
                    }
                } else {
                    for (int32_t offset : nan_offsets) {
                        this->update1_nan(offset);
                    }
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

    // TODO: do we still need these?
    value_type update1_nan(int64_t index = 0) {
        this->nan_count++;
        return static_cast<Derived &>(*this).add_nan(index);
    }

    py::object key_array() {
        py::array_t<key_type> output_array(this->length());
        auto output = output_array.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        auto offsets = this->offsets();
        size_t map_index = 0;
        int64_t natural_order = 0;
        // TODO: can be parallel due to non-overlapping maps
        for (auto &map : this->maps) {
            for (auto &el : map) {
                key_type key = el.first;
                int64_t index = static_cast<Derived &>(*this).key_offset(natural_order++, map_index, el, offsets[map_index]);
                output(index) = key;
            }
            map_index += 1;
        }
        if (this->nan_count) {
            output(this->nan_index()) = NaNish<key_type>::value;
        }
        if (this->null_count) {
            output(this->null_index()) = -1;
        }
        return output_array;
    }

    std::vector<std::map<key_type, value_type>> extract() {
        std::vector<std::map<key_type, value_type>> map_vector;
        for (auto &map : this->maps) {
            std::map<key_type, value_type> m;
            for (auto &el : map) {
                key_type value = el.first;
                m[value] = el.second;
            }
            map_vector.push_back(std::move(m));
        }
        return map_vector;
    }
};

template <class U, template <typename, typename> class Hashmap2>
class counter : public hash_base<counter<U, Hashmap2>, U, Hashmap2>, public counter_mixin<U, int64_t, counter<U, Hashmap2>> {
  public:
    using Base = hash_base<counter<U, Hashmap2>, U, Hashmap2>;
    using typename Base::hashmap_type;
    using typename Base::key_type;
    using typename Base::value_type;

    counter(int nmaps = 1) : Base(nmaps) {}

    template <class Bucket>
    int64_t key_offset(int64_t natural_order, int16_t map_index, Bucket &bucket, int64_t offset) {
        int64_t effective_offset = natural_order;
        if (this->null_count) {
            effective_offset += 1;
        }
        if (this->nan_count) {
            effective_offset += 1;
        }
        return effective_offset;
    }
    value_type add_null(int64_t index) {
        // parent already keeps track of the counts
        return this->null_count;
    }
    value_type add_nan(int64_t index) {
        // same
        return this->nan_count;
    }
    int64_t value_null() { return this->null_count; }
    int64_t value_nan() { return this->nan_count; }
    value_type add_new(int16_t map_index, key_type &value, int64_t index) {
        auto &map = this->maps[map_index];
        map.emplace(value, 1);
        return 1;
    }
    template <class Bucket>
    value_type add_existing(Bucket &bucket, int16_t map_index, key_type &value, int64_t index) {
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
                // key_type key = el.first;
                value_type value = el.second;
                int64_t index = key_offset(natural_order++, map_index, el, offsets[map_index]);
                output(index) = value;
            }
            map_index += 1;
        }
        if (this->nan_count) {
            output(this->nan_index()) = this->nan_count;
        }
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
            for (auto &elem : other.maps[i]) {
                const key_type &value = elem.first;
                auto search = this->maps[i].find(value);
                auto end = this->maps[i].end();
                if (search == end) {
                    this->maps[i].emplace(elem);
                } else {
                    set_second(search, search->second + elem.second);
                }
            }
        }
        this->nan_count += other.nan_count;
        this->null_count += other.null_count;
    }
};

template <class T2, template <typename, typename> class Hashmap2>
class ordered_set : public hash_base<ordered_set<T2, Hashmap2>, T2, Hashmap2> {
  public:
    using Base = hash_base<ordered_set<T2, Hashmap2>, T2, Hashmap2>;
    using typename Base::hasher_map_choice;
    using typename Base::hashmap_type;
    using typename Base::key_type;
    using typename Base::value_type;

    ordered_set(int nmaps = 1, int64_t limit = -1) : Base(nmaps, limit), nan_value(0x7fffffff), null_value(0x7fffffff), ordinal_code_offset_null_nan(0) {}

    virtual value_type nan_index() { return nan_value; }
    virtual value_type null_index() { return null_value; }
    template <class Bucket>
    int64_t key_offset(int64_t natural_order, int16_t map_index, Bucket &bucket, int64_t offset) {
        int64_t index = bucket.second + offset;
        return index;
    }

    value_type add_nan(int64_t index) {
        // first time we add it
        if (this->nan_count == 1) {
            nan_value = this->maps[0].size() + ordinal_code_offset_null_nan;
            ordinal_code_offset_null_nan++;
        }
        return nan_value;
    }
    value_type add_null(int64_t index) {
        if (this->null_count == 1) {
            null_value = this->maps[0].size() + ordinal_code_offset_null_nan;
            ordinal_code_offset_null_nan++;
        }
        return null_value;
    }
    int64_t value_null() { return (this->nan_count > 0) ? 1 : 0; }
    int64_t value_nan() { return 0; }
    value_type add_new(int16_t map_index, key_type &value, int64_t index) {
        auto &map = this->maps[map_index];
        value_type ordinal_code = map.size();
        if (map_index == 0) {
            ordinal_code += ordinal_code_offset_null_nan;
        }
        map.emplace(value, ordinal_code);
        return ordinal_code;
    }
    template <class Bucket>
    value_type add_existing(Bucket &bucket, int16_t map_index, key_type &value, int64_t index) {
        // we can do nothing here
        return bucket->second;
    }

    static ordered_set *create(py::array_t<key_type> keys, int64_t null_value, int64_t nan_count, int64_t null_count, std::string* fingerprint) {
        ordered_set *set = new ordered_set(1);
        const key_type *keys_ptr = keys.data(0);
        size_t size = keys.size();
        {
            py::gil_scoped_release gil;
            for (size_t i = 0; i < size; i++) {
                key_type key = keys_ptr[i];
                if (int64_t(i) == null_value) {
                    set->update1_null();
                } else if (custom_isnan(key)) {
                    set->update1_nan();
                } else {
                    set->update1(0, key);
                }
            }
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
        if (set->count() != keys.size()) {
            throw std::runtime_error(std::string("key array of length ") + std::to_string(keys.size()) + " does not match expected length of " + std::to_string(set->count()));
        }
        set->null_count = null_count;
        set->nan_count = nan_count;
        set->sealed = true;
        if(fingerprint) {
            set->fingerprint = *fingerprint;
        }
        return set;
    }

    py::object isin(py::array_t<key_type> &values) {
        int64_t size = values.size();
        py::array_t<bool> result(size);
        auto input = values.template unchecked<1>();
        auto output = result.template mutable_unchecked<1>();
        size_t nmaps = this->maps.size();
        py::gil_scoped_release gil;
        for (int64_t i = 0; i < size; i++) {
            const key_type &value = input(i);
            if (custom_isnan(value)) {
                output(i) = this->nan_count > 0;
            } else {
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
    py::object map_ordinal(py::array_t<key_type> &values) {
        size_t size = this->length();
        // TODO: apply this pattern of various return types to the other set types
        if (size < (1u << 7u)) {
            return this->template _map_ordinal<int8_t>(values);
        } else if (size < (1u << 15u)) {
            return this->template _map_ordinal<int16_t>(values);
        } else if (size < (1u << 31u)) {
            return this->template _map_ordinal<int32_t>(values);
        } else {
            return this->template _map_ordinal<int64_t>(values);
        }
    }
    template <class OutputType>
    py::array_t<OutputType> _map_ordinal(py::array_t<key_type> &values) {
        int64_t size = values.size();
        py::array_t<OutputType> result(size);
        auto input = values.template unchecked<1>();
        auto output = result.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        size_t nmaps = this->maps.size();
        auto offsets = this->offsets();
        for (int64_t i = 0; i < size; i++) {
            const key_type &value = input(i);
            // the caller is responsible for finding masked values
            if (custom_isnan(value)) {
                output(i) = this->nan_value;
                assert(this->nan_count > 0);
            } else {
                std::size_t hash = hasher_map_choice()(value);
                size_t map_index = (hash % nmaps);
                auto search = this->maps[map_index].find(value);
                if (search == this->maps[map_index].end()) {
                    output(i) = -1;
                } else {
                    output(i) = search->second + offsets[map_index];
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
            for (size_t i = 0; i < this->maps.size(); i++) {
                for (auto &elem : other->maps[i]) {
                    const key_type &value = elem.first;
                    auto search = this->maps[i].find(value);
                    auto end = this->maps[i].end();
                    if (search == end) {
                        this->maps[i].emplace(value, this->maps[i].size());
                    } else {
                        // if already in, it's fine
                    }
                }
                other->maps[i].clear();
            }
            this->nan_count += other->nan_count;
            this->null_count += other->null_count;
        }
    }

    value_type nan_value;
    value_type null_value;
    value_type ordinal_code_offset_null_nan;
    // py::object value_array() {
    //     py::object np = py::module::import("numpy");
    //     return np.attr("arange")(0, this->length(), 1, "int64");
    // }
};

template <class T2, template <typename, typename> class Hashmap2>
class index_hash : public hash_base<index_hash<T2, Hashmap2>, T2, Hashmap2> {
  public:
    using Base = hash_base<index_hash<T2, Hashmap2>, T2, Hashmap2>;
    using typename Base::hasher_map_choice;
    using typename Base::hashmap_type;
    using typename Base::key_type;
    using typename Base::value_type;

    // TODO: might be better to use a node based hasmap, we don't want to move large vectors
    typedef hashmap<key_type, std::vector<int64_t>> overflow_type;
    index_hash(int nmaps = 1) : Base(nmaps), overflows(nmaps), has_duplicates(false) {}

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
    int64_t key_offset(int64_t natural_order, int16_t map_index, Bucket &bucket, int64_t offset) {
        return natural_order;
    }

    value_type add_nan(int64_t index) {
        this->nan_value = index;
        return this->nan_value;
    }

    value_type add_null(int64_t index) {
        this->null_value = index;
        return this->null_value;
    }

    value_type add_new(int16_t map_index, key_type &value, int64_t index) {
        auto &map = this->maps[map_index];
        map.emplace(value, index);
        return index;
    }

    template <class Bucket>
    value_type add_existing(Bucket &position, int16_t map_index, key_type &value, int64_t index) {
        // we found a duplicate
        overflows[map_index][position->first].push_back(index);
        has_duplicates = true;
        return index;
    }

    int64_t value_null() { return this->null_value; }
    int64_t value_nan() { return this->nan_value; }

    py::array_t<value_type> map_index(py::array_t<key_type, py::array::c_style> &values) {
        value_type size = values.size();
        py::array_t<value_type, py::array::c_style> result(size);
        map_index_write(values, result);
        return result;
    }

    template <typename result_type>
    bool map_index_write(py::array_t<key_type, py::array::c_style> &values, py::array_t<result_type, py::array::c_style> &output_array) {
        int64_t size = values.size();
        auto input = values.template unchecked<1>();
        auto output = output_array.template mutable_unchecked<1>();
        bool encountered_unknown = false;
        int16_t nmaps = this->maps.size();
        py::gil_scoped_release gil;

        for (int64_t i = 0; i < size; i++) {
            const key_type &key = input(i);
            if (custom_isnan(key)) {
                output(i) = nan_value;
                assert(this->nan_count > 0);
            } else {
                std::size_t hash = hasher_map_choice()(key);
                size_t map_index = (hash % nmaps);
                auto search = this->maps[map_index].find(key);
                auto end = this->maps[map_index].end();
                if (search == end) {
                    output(i) = -1;
                    encountered_unknown = true;
                } else {
                    output(i) = search->second;
                }
            }
        }
        return encountered_unknown;
    }
    py::array_t<value_type> map_index_with_mask(py::array_t<key_type, py::array::c_style> &values, py::array_t<uint8_t, py::array::c_style> &mask) {
        value_type size = values.size();
        py::array_t<value_type, py::array::c_style> result(size);
        map_index_with_mask_write(values, mask, result);
        return result;
    }
    template <typename result_type>
    bool map_index_with_mask_write(py::array_t<key_type, py::array::c_style> &values, py::array_t<uint8_t, py::array::c_style> &mask, py::array_t<result_type, py::array::c_style> &output_array) {
        int64_t size = values.size();
        assert(values.size() == mask.size());
        auto input = values.template unchecked<1>();
        auto input_mask = mask.template unchecked<1>();
        auto output = output_array.template mutable_unchecked<1>();
        bool encountered_unknown = false;
        int16_t nmaps = this->maps.size();
        py::gil_scoped_release gil;

        for (int64_t i = 0; i < size; i++) {
            const key_type &key = input(i);
            if (custom_isnan(key)) {
                output(i) = nan_value;
                assert(this->nan_count > 0);
            } else if (input_mask(i) == 1) {
                output(i) = null_value;
                assert(this->null_count > 0);
            } else {
                std::size_t hash = hasher_map_choice()(key);
                size_t map_index = (hash % nmaps);
                auto search = this->maps[map_index].find(key);
                auto end = this->maps[map_index].end();
                if (search == end) {
                    output(i) = -1;
                    encountered_unknown = true;
                } else {
                    output(i) = search->second;
                }
            }
        }
        return encountered_unknown;
    }

    std::tuple<py::array_t<value_type>, py::array_t<value_type>> map_index_duplicates_with_mask(py::array_t<key_type> &values, py::array_t<uint8_t> &mask, int64_t start_index) {
        std::vector<typename overflow_type::value_type> found; // should this be a reference to the key_type?
        std::vector<value_type> indices;
        size_t size = values.size();
        size_t size_output = 0;
        int16_t nmaps = this->maps.size();
        auto input = values.template unchecked<1>();
        auto input_mask = mask.template unchecked<1>();

        {
            py::gil_scoped_release gil;
            for (size_t i = 0; i < size; i++) {
                const key_type &key = input(i);
                if (custom_isnan(key)) {
                } else if (input_mask(i) == 1) {
                } else {
                    std::size_t hash = hasher_map_choice()(key);
                    size_t map_index = (hash % nmaps);
                    auto search = this->overflows[map_index].find(key);
                    auto end = this->overflows[map_index].end();
                    if (search != end) {
                        found.push_back(search->first);
                        size_output += search->second.size();
                        indices.insert(indices.end(), search->second.size(), start_index + i);
                    }
                }
            }
        }

        py::array_t<value_type> result(size_output);
        py::array_t<value_type> indices_array(size_output);
        auto output = result.template mutable_unchecked<1>();
        auto output_indices = indices_array.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        // int64_t offset = 0;
        size_t index = 0;

        std::copy(indices.begin(), indices.end(), &output_indices(0));

        for (auto &el : found) {
            std::vector<value_type> &indices = el.second;
            for (int64_t i : indices) {
                output(index++) = i;
            }
        }
        return std::make_tuple(indices_array, result);
    }

    std::tuple<py::array_t<value_type>, py::array_t<value_type>> map_index_duplicates(py::array_t<key_type> &values, int64_t start_index) {
        std::vector<typename overflow_type::value_type> found; // should this be a reference to the key_type?
        std::vector<value_type> indices;
        size_t size = values.size();
        size_t size_output = 0;
        int16_t nmaps = this->maps.size();
        auto input = values.template unchecked<1>();

        {
            py::gil_scoped_release gil;
            for (size_t i = 0; i < size; i++) {
                const key_type &key = input(i);
                if (custom_isnan(key)) {
                } else {
                    std::size_t hash = hasher_map_choice()(key);
                    size_t map_index = (hash % nmaps);
                    auto search = this->overflows[map_index].find(key);
                    auto end = this->overflows[map_index].end();
                    if (search != end) {
                        found.push_back(*search);
                        size_output += search->second.size();
                        indices.insert(indices.end(), search->second.size(), start_index + i);
                    }
                }
            }
        }

        py::array_t<value_type> result(size_output);
        py::array_t<value_type> indices_array(size_output);
        auto output = result.template mutable_unchecked<1>();
        auto output_indices = indices_array.template mutable_unchecked<1>();
        py::gil_scoped_release gil;
        // int64_t offset = 0;
        size_t index = 0;

        std::copy(indices.begin(), indices.end(), &output_indices(0));

        for (auto &el : found) {
            std::vector<value_type> &indices = el.second;
            for (int64_t i : indices) {
                output(index++) = i;
            }
        }
        return std::make_tuple(indices_array, result);
    }

    void merge(const index_hash &other) {
        py::gil_scoped_release gil;
        if (this->maps.size() != other.maps.size()) {
            throw std::runtime_error("cannot merge with an unequal maps");
        }

        for (size_t i = 0; i < this->maps.size(); i++) {
            for (auto &elem : other.maps[i]) {
                const key_type &key = elem.first;
                auto search = this->maps[i].find(key);
                auto end = this->maps[i].end();
                if (search == end) {
                    this->maps[i].emplace(key, elem.second);
                } else {
                    // if already in, add it to the multimap
                    overflows[i][elem.first].push_back(elem.second);
                }
            }
        }
        this->nan_count += other.nan_count;
        this->null_count += other.null_count;
        for (size_t i = 0; i < this->maps.size(); i++) {
            for (auto el : other.overflows[i]) {
                std::vector<int64_t> &source = el.second;

                const key_type &key = el.first;
                // const value_type& value = elem.first;
                auto search = this->maps[i].find(key);
                auto end = this->maps[i].end();
                if (search == end) {
                    // we have a duplicate that is not in the current map, so we insert the first element
                    this->maps[i].emplace(key, source[0]);
                    if (source.size() > 1) {
                        std::vector<int64_t> &target = this->overflows[i][key];
                        target.insert(target.end(), source.begin() + 1, source.end());
                    }
                } else {
                    // easy case, just merge the vectors
                    std::vector<int64_t> &target = this->overflows[i][key];
                    target.insert(target.end(), source.begin(), source.end());
                }
            }
        }
        has_duplicates = has_duplicates || other.has_duplicates;
    }

    std::vector<overflow_type> overflows; // this stores only the duplicates
    int64_t null_value;
    int64_t nan_value;
    bool has_duplicates;
};
} // namespace vaex

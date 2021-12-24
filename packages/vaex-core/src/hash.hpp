// #include "flat_hash_map.hpp"
// #include "unordered_map.hpp"
// #define VAEX_USE_TSL 1
// #define VAEX_USE_ABSL 1

#ifdef VAEX_USE_TSL
#include "tsl/hopscotch_map.h"
#include "tsl/hopscotch_set.h"
#endif

#ifdef VAEX_USE_ABSL
#include "absl/container/flat_hash_map.h"
#endif

#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace vaex {

namespace py = pybind11;

// simple and fast https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static inline std::size_t _hash64(uint64_t x) {
    x = (x ^ (x >> 30)) * uint64_t(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * uint64_t(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
}

// 64 and 32 bit should not have identity as hash function, if the lower bits are not used (like with Gaia's source_id)
// we get horrible performance, which can be alivated by using prime growth

template <typename T>
struct hash {
    std::size_t operator()(T const &val) const {
        std::hash<T> h;
        return h(val);
    }
};

template <>
struct hash<uint64_t> {
    std::size_t operator()(uint64_t const val) const { return _hash64(val); }
};

template <>
struct hash<int64_t> {
    std::size_t operator()(uint64_t const val) const { return _hash64(*reinterpret_cast<const int64_t *>(&val)); }
};

template <>
struct hash<int32_t> {
    std::size_t operator()(int32_t const val) const {
        uint64_t v(val);
        return _hash64(v);
    }
};

template <>
struct hash<uint32_t> {
    std::size_t operator()(uint32_t const val) const {
        uint64_t v(val);
        return _hash64(v);
    }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
template <>
struct hash<float> {
    std::size_t operator()(float const val) const {
        uint64_t v(0);
        *reinterpret_cast<float *>(&v) = val;
        return _hash64(v);
    }
};

template <>
struct hash<double> {
    std::size_t operator()(double const val) const { return _hash64(*reinterpret_cast<const uint64_t *>(&val)); }
};
#pragma GCC diagnostic pop

#ifdef VAEX_USE_TSL
template <class Key, class Value, class Hash = vaex::hash<Key>, class Compare = std::equal_to<Key>>
using hashmap = tsl::hopscotch_map<Key, Value, Hash, Compare>;
template <class Key, class Value, class Hash = vaex::hash<Key>, class Compare = std::equal_to<Key>>
using hashmap_pg = tsl::hopscotch_pg_map<Key, Value, Hash, Compare>;

template <class Key, class Value, class Hash = vaex::hash<Key>, class Compare = std::equal_to<Key>>
using hashmap_store = tsl::hopscotch_map<Key, Value, Hash, Compare, std::allocator<std::pair<Key, Value>>, 30, true>;
template <class Key, class Value, class Hash = vaex::hash<Key>, class Compare = std::equal_to<Key>>
using hashmap_pg_store = tsl::hopscotch_pg_map<Key, Value, Hash, Compare, std::allocator<std::pair<Key, Value>>, 30, true>;
#endif

#ifdef VAEX_USE_ABSL
template <class Key, class Value, class Hash = vaex::hash<Key>, class Compare = std::equal_to<Key>>
using hashmap = absl::flat_hash_map<Key, Value, Hash, Compare>;
template <class Key, class Value, class Hash = vaex::hash<Key>, class Compare = std::equal_to<Key>>
using hashmap_pg = absl::flat_hash_map<Key, Value, Hash, Compare>;

template <class Key, class Value, class Hash = vaex::hash<Key>, class Compare = std::equal_to<Key>>
using hashmap_store = absl::flat_hash_map<Key, Value, Hash, Compare>;
template <class Key, class Value, class Hash = vaex::hash<Key>, class Compare = std::equal_to<Key>>
using hashmap_pg_store = absl::flat_hash_map<Key, Value, Hash, Compare>;
#endif

// template<class Key,  class Hash, class Compare>
// using hashset = tsl::hopscotch_set<Key, Hash, Compare>;

// we cannot modify .second, instead use .value()
// see https://github.com/Tessil/hopscotch-map
template <class I, class V>
inline void set_second(I &it, V &&value) {
#ifdef VAEX_USE_TSL
    it.value() = value;
#else
    it->second = value;
#endif
}

template <class Derived, class KeyType, class Hashmap = hashmap<KeyType, int64_t>>
class hash_common {
  public:
    using value_type = int64_t;
    using key_type = KeyType;
    using hashmap_type = Hashmap;
    using hasher = typename hashmap_type::hasher;
    using hasher_map = typename hashmap_type::hasher;
    using hasher_map_choice = typename hashmap_type::hasher; // typename vaex::hash<key_type>;

    hash_common(int16_t nmaps, int64_t limit = -1) : maps(nmaps), limit(limit), maplocks(nmaps), nan_count(0), null_count(0), sealed(false) {}

    void update1(key_type &key) {
        std::size_t hash = hasher_map_choice()(key);
        size_t map_index = (hash % this->maps.size());
        update1(map_index, key);
    }

    void update1(uint16_t map_index, key_type &key, int64_t index = 0) {
        auto &map = this->maps[map_index];
        auto search = map.find(key);
        auto end = map.end();
        if (search == end) {
            static_cast<Derived &>(*this).add_new(map_index, key, index);
        } else {
            static_cast<Derived &>(*this).add_existing(search, map_index, key, index);
        }
    }
    value_type update1_null(int64_t index = 0) {
        this->null_count++;
        return static_cast<Derived &>(*this).add_null(index);
    }

    py::object flatten_values(py::array_t<value_type> values, py::array_t<int16_t> map_index, py::array_t<value_type> out) {
        int64_t size = values.size();
        if (values.size() != out.size()) {
            throw std::runtime_error("output array does not match length of values");
        }
        if (values.size() != map_index.size()) {
            throw std::runtime_error("map_index array does not match length of values");
        }
        auto result_ptr = out.template mutable_unchecked<1>();
        auto out_ptr = values.template unchecked<1>();
        auto map_index_ptr = map_index.template unchecked<1>();
        std::vector<int64_t> offsets = this->offsets();
        {
            py::gil_scoped_release gil;
            for (int64_t i = 0; i < size; i++) {
                result_ptr(i) = out_ptr(i) + offsets[map_index_ptr(i)];
            }
        }
        return std::move(out);
    }

    virtual value_type nan_index() { return 0; }
    virtual value_type null_index() { return this->nan_count ? 1 : 0; }
    py::list keys() {
        // if(!this->sealed) {
        //     throw std::runtime_error("hashmap not sealed, call .seal() first");
        // }
        py::list l(this->length());
        size_t map_index = 0;
        int64_t natural_order = 0;
        auto offsets = this->offsets();
        for (auto &map : this->maps) {
            for (auto &el : map) {
                key_type key = el.first;
                int64_t index = static_cast<Derived &>(*this).key_offset(natural_order++, map_index, el, offsets[map_index]);
                l[index] = key;
            }
            map_index += 1;
        }
        if (this->nan_count) {
            py::object math = py::module::import("math");
            l[nan_index()] = math.attr("nan");
        }
        if (this->null_count) {
            l[null_index()] = py::none();
        }
        return l;
    }

    void seal() { sealed = true; }

    int64_t count() const {
        int64_t c = 0;
        for (size_t i = 0; i < this->maps.size(); i++) {
            c += this->maps[i].size();
            if (i == 0) {
                if (this->null_count) {
                    c++;
                }
                if (this->nan_count) {
                    c++;
                }
            }
        }
        return c;
    }

    std::vector<int64_t> offsets() const {
        std::vector<int64_t> v;
        int64_t offset = 0;
        for (size_t i = 0; i < this->maps.size(); i++) {
            v.push_back(offset);
            offset += this->maps[i].size();
            if (i == 0) {
                if (this->null_count) {
                    offset++;
                }
                if (this->nan_count) {
                    offset++;
                }
            }
        }
        return v;
    }

    int64_t offset() const {
        // null and nan are special cases, and always in the begin
        return (this->null_count > 0 ? 1 : 0) + (this->nan_count > 0 ? 1 : 0);
    }
    int64_t length() const {
        // normal count and null and nan
        return count();
    }

    std::vector<hashmap_type> maps;
    int64_t limit;
    std::vector<std::mutex> maplocks;
    int64_t nan_count;
    int64_t null_count;
    bool sealed;
    std::string fingerprint;
};

template <class K, class V, class Derived>
class counter_mixin {
  public:
    using value_type = V;
    using key_type = K;

    // ideally, we put common code in mixin classes using CRTP, but this cannot be resolved?
    // value_type add(int16_t map_index, key_type& value, int64_t index) {
    //     auto & map = static_cast<Derived&>(*this)->maps[map_index];
    //     map.emplace(value, 1);
    //     return 1;
    // }
};

} // namespace vaex

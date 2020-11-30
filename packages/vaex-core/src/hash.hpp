// #include "flat_hash_map.hpp"
// #include "unordered_map.hpp"
#include "tsl/hopscotch_set.h"
#include "tsl/hopscotch_map.h"

namespace vaex {

// simple and fast https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static inline std::size_t _hash64(uint64_t x) {
    x = (x ^ (x >> 30)) * uint64_t(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * uint64_t(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
}

// 64 and 32 bit should not have identity as hash function, if the lower bits are not used (like with Gaia's source_id)
// we get horrible performance, which can be alivated by using prime growth

template<typename T>
struct hash {
    std::size_t operator()(T const& val) const {
        std::hash<T> h;
        return h(val);
    }
};

template<>
struct hash<uint64_t> {
    std::size_t operator()(uint64_t const val) const {
        return _hash64(val);
    }
};

template<>
struct hash<int64_t> {
    std::size_t operator()(uint64_t const val) const {
        return _hash64(*reinterpret_cast<const int64_t*>(&val));
    }
};


template<>
struct hash<int32_t> {
    std::size_t operator()(int32_t const val) const {
        uint64_t v(val);
        return _hash64(v);
    }
};

template<>
struct hash<uint32_t> {
    std::size_t operator()(uint32_t const val) const {
        uint64_t v(val);
        return _hash64(v);
    }
};

template<>
struct hash<float> {
    std::size_t operator()(float const val) const {
        uint64_t v(0);
        *reinterpret_cast<float*>(&v) = val;
        return _hash64(v);
    }
};

template<>
struct hash<double> {
    std::size_t operator()(double const val) const {
        return _hash64(*reinterpret_cast<const uint64_t*>(&val));
    }
};


template<class Key, class Value, class Hash=vaex::hash<Key>, class Compare=std::equal_to<Key>>
// using hashmap = ska::flat_hash_map<Key, Value, Hash, Compare>;
using hashmap = tsl::hopscotch_map<Key, Value, Hash, Compare>;
template<class Key, class Value, class Hash=vaex::hash<Key>, class Compare=std::equal_to<Key>>
using hashmap_pg = tsl::hopscotch_pg_map<Key, Value, Hash, Compare>;
// template<class Key,  class Hash, class Compare>
// using hashset = tsl::hopscotch_set<Key, Hash, Compare>;

// we cannot modify .second, instead use .value()
// see https://github.com/Tessil/hopscotch-map
template<class I, class V>
inline void set_second(I& it, V &&value) {
    it.value() = value;
}

}

// #include "flat_hash_map.hpp"
// #include "unordered_map.hpp"
#include "tsl/hopscotch_set.h"
#include "tsl/hopscotch_map.h"

namespace vaex {

template<class Key, class Value, class Hash=std::hash<Key>, class Compare=std::equal_to<Key>>
// using hashmap = ska::flat_hash_map<Key, Value, Hash, Compare>;
using hashmap = tsl::hopscotch_map<Key, Value, Hash, Compare>;
// template<class Key,  class Hash, class Compare>
// using hashset = tsl::hopscotch_set<Key, Hash, Compare>;

// we cannot modify .second, instead use .value()
// see https://github.com/Tessil/hopscotch-map
template<class I, class V>
inline void set_second(I& it, V &&value) {
    it.value() = value;
}

}

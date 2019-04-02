#include "flat_hash_map.hpp"
#include "unordered_map.hpp"

namespace vaex {

template<class K, class V>
using hashmap = ska::flat_hash_map<K, V>;
// using hashmap = tsl::hopscotch_map<K, V>;

}
#include "hash_primitives.cpp"

namespace vaex {
template<class T, class M>
void init_hash_pg(M m, std::string name) {
    init_hash_<T, M, hashmap_primitive_pg>(m, name, "_prime_growth");
}

void init_hash_primitives_prime(py::module &m) {
    init_hash_pg<int64_t>(m, "int64");
    init_hash_pg<uint64_t>(m, "uint64");
    init_hash_pg<int32_t>(m, "int32");
    init_hash_pg<uint32_t>(m, "uint32");
    init_hash_pg<int16_t>(m, "int16");
    init_hash_pg<uint16_t>(m, "uint16");
    init_hash_pg<int8_t>(m, "int8");
    init_hash_pg<uint8_t>(m, "uint8");
    init_hash_pg<bool>(m, "bool");
    init_hash_pg<float>(m, "float32");
    init_hash_pg<double>(m, "float64");
}
};


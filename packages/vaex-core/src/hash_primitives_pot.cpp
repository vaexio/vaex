#include "hash_primitives.cpp"

namespace vaex {
template<class T, class M>
void init_hash(M m, std::string name) {
    init_hash_<T, M, hashmap_primitive>(m, name, "");
}

void init_hash_primitives_power_of_two(py::module &m) {
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
    init_hash<double>(m, "float64");
}
};
#include "hash_string.hpp"

// #define VAEX_HASH_STRING_SIMPLE

namespace vaex {

// cannot get stringview to work with msvc
// struct equal_string {
//     using is_transparent = void;

//     bool operator()(const string &str, const string_view &strview) const { return str == strview; }

//     bool operator()(const string_view &strview1, const string_view &strview2) const { return strview1 == strview2; }

//     bool operator()(const string_view strview, const string &str) const { return strview == str; }

//     bool operator()(const string &str1, const string &str2) const { return str1 == str2; }
// };

// struct hash_string {
//     std::size_t operator()(const string &str) const {
// #ifdef VAEX_HASH_STRING_SIMPLE
//         unsigned int hash = 1;
//         const char *s = str.data();
//         const char *end = s + str.size();
//         while (s != end) {
//             hash = hash * 101 + *s++;
//         }
//         return hash;
// #else
//         return std::hash<string>()(str);
// #endif
//     }

//     std::size_t operator()(const string_view str_view) const {
// #ifdef VAEX_HASH_STRING_SIMPLE
//         unsigned int hash = 1;
//         const char *s = str_view.data();
//         const char *end = s + str_view.size();
//         while (s != end) {
//             hash = hash * 101 + *s++;
//         }
//         return hash;
// #else
//         return std::hash<string_view>()(str_view);
// #endif
//     }
// };

template <class Type, class Cls>
void bind_common(Cls &cls) {
    cls.def("update", &Type::update, "add values", py::arg("values"), py::arg("start_index") = 0, py::arg("chunk_size") = 1024 * 128, py::arg("bucket_size") = 1024 * 128,
            py::arg("return_values") = false)
        .def("seal", &Type::seal)
        .def("merge", &Type::merge)
        .def("extract", &Type::extract)
        .def("keys", &Type::keys)
        .def("key_array", &Type::key_array)
        .def("offsets", &Type::offsets)
        .def_property_readonly("count", &Type::count)
        .def("__len__", &Type::length)
        .def("__sizeof__", &Type::bytes_used)
        .def_property_readonly("offset", &Type::offset)
        .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
        .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
        .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
        .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; });
}

void init_hash_string(py::module &m) {
    {
        typedef counter<> Type;
        std::string countername = "counter_string";
        auto cls = py::class_<Type>(m, countername.c_str()).def(py::init<int>()).def("counts", &Type::counts);
        bind_common<Type>(cls);
    }
    {
        std::string ordered_setname = "ordered_set_string";
        typedef ordered_set<> Type;
        auto cls = py::class_<Type>(m, ordered_setname.c_str())
                       .def(py::init<int, int64_t>(), py::arg("nmaps"), py::arg("limit") = -1)
                       .def(py::init(&Type::create<StringList32>))
                       .def(py::init(&Type::create<StringList64>))
                       .def("isin", &Type::isin)
                       .def("flatten_values", &Type::flatten_values)
                       .def("map_ordinal", &Type::map_ordinal)
                       .def_property_readonly("null_value", [](const Type &c) { return c.null_value; })
                       .def_readwrite("fingerprint", &Type::fingerprint);
        bind_common<Type>(cls);
    }
    {
        std::string index_hashname = "index_hash_string";
        typedef index_hash<> Type;
        auto cls = py::class_<Type>(m, index_hashname.c_str())
                       .def(py::init<int>())
                       .def("map_index", &Type::map_index)
                       .def("map_index", &Type::template map_index_write<int8_t>)
                       .def("map_index", &Type::template map_index_write<int16_t>)
                       .def("map_index", &Type::template map_index_write<int32_t>)
                       .def("map_index", &Type::template map_index_write<int64_t>)
                       .def("map_index_duplicates", &Type::map_index_duplicates)
                       .def_property_readonly("has_duplicates", [](const Type &c) { return c.has_duplicates; })
            // .def_property_readonly("overflow_size", [](const Type &c) { return c.map.overflow_size(); })
            ;
        bind_common<Type>(cls);
    }
}
} // namespace vaex

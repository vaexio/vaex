#include "hash_string.hpp"

// #define VAEX_HASH_STRING_SIMPLE

namespace vaex {

// cannot get stringview to work with msvc
// struct equal_string {
//     using is_transparent = void;

//     bool operator()(const string& str, const string_view& strview) const {
//         return str == strview;
//     }

//     bool operator()(const string_view& strview1, const string_view& strview2) const {
//         return strview1 == strview2;
//     }

//     bool operator()(const string_view strview, const string& str) const {
//         return strview == str;
//     }

//     bool operator()(const string& str1, const string& str2) const {
//         return str1 == str2;
//     }
// };

// struct hash_string {
//     std::size_t operator()(const string& str) const {
//         #ifdef VAEX_HASH_STRING_SIMPLE
//             unsigned int hash = 1;
//             const char *s = str.data();
//             const char *end = s + str.size();
//             while(s != end) {
//                 hash = hash * 101  +  *s++;
//             }
//             return hash;
//         #else
//             return std::hash<string>()(str);
//         #endif
//     }

//     std::size_t operator()(const string_view str_view) const {
//         #ifdef VAEX_HASH_STRING_SIMPLE
//             unsigned int hash = 1;
//             const char *s = str_view.data();
//             const char *end = s + str_view.size();
//             while(s != end) {
//                 hash = hash * 101  +  *s++;
//             }
//             return hash;
//         #else
//             return std::hash<string_view>()(str_view);
//         #endif
//     }
// };


void init_hash_string(py::module &m) {
    {
        typedef counter<> counter_type;
        std::string countername = "counter_string";
        py::class_<counter_type>(m, countername.c_str())
            .def(py::init<>())
            .def("update", &counter_type::update, "add values", py::arg("values"), py::arg("start_index") = 0)
            .def("merge", &counter_type::merge)
            .def("extract", &counter_type::extract)
            .def_property_readonly("count", [](const counter_type &c) { return c.count; })
            .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const counter_type &c) { return c.null_count; })
        ;
    }
    // {
    //     typedef counter<string_view, string_view, string_view> counter_type;
    //     std::string countername = "counter_stringview";
    //     py::class_<counter_type>(m, countername.c_str())
    //         .def(py::init<>())
    //         .def("update", &counter_type::update)
    //         .def("merge", &counter_type::merge)
    //         .def("extract", &counter_type::extract)
    //         .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
    //         .def_property_readonly("null_count", [](const counter_type &c) { return c.null_count; })
    //     ;
    // }
    {
        std::string ordered_setname = "ordered_set_string";
        typedef ordered_set<> Type;
        py::class_<Type>(m, ordered_setname.c_str())
            .def(py::init<>())
            .def(py::init(&Type::create))
            .def("isin", &Type::isin)
            .def("update", &Type::update, "add values", py::arg("values"), py::arg("start_index") = 0)
            // .def("update", &Type::update_with_mask)
            .def("merge", &Type::merge)
            .def("extract", &Type::extract)
            .def("keys", &Type::keys)
            .def("map_ordinal", &Type::map_ordinal)
            .def_property_readonly("count", [](const Type &c) { return c.count; })
            .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
            .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
            .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
        ;
    }
     {
        std::string index_hashname = "index_hash_string";
        typedef index_hash<> Type;
        py::class_<Type>(m, index_hashname.c_str())
            .def(py::init<>())
            .def("update", &Type::update)
            // .def("update", &Type::update_with_mask)
            .def("merge", &Type::merge)
            .def("extract", &Type::extract)
            .def("keys", &Type::keys)
            .def("map_index", &Type::map_index)
            .def("map_index", &Type::template map_index_write<int8_t>)
            .def("map_index", &Type::template map_index_write<int16_t>)
            .def("map_index", &Type::template map_index_write<int32_t>)
            .def("map_index", &Type::template map_index_write<int64_t>)
            .def("map_index_duplicates", &Type::map_index_duplicates)
            .def("__len__", [](const Type &c) { return c.count + (c.null_count > 0) + (c.nan_count > 0); })
            .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
            .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
            .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
            .def_property_readonly("has_duplicates", [](const Type &c) { return c.has_duplicates; })
        ;
    }
    // {
    //     std::string ordered_setname = "ordered_set_stringview";
    //     typedef ordered_set<string_view, string_view> Type;
    //     py::class_<Type>(m, ordered_setname.c_str())
    //         .def(py::init<>())
    //         .def("update", &Type::update)
    //         // .def("update", &Type::update_with_mask)
    //         .def("merge", &Type::merge)
    //         .def("extract", &Type::extract)
    //         .def("keys", &Type::keys)
    //         .def("map_ordinal", &Type::map_ordinal)
    //         .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
    //         .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
    //         .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
    //         .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
    //     ;
    // }
}
}

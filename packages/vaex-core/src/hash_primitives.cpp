#include "hash_primitives.hpp"

template <class Type, class Cls>
void bind_common(Cls &cls) {
    cls.def("update", &Type::update, "add values", py::arg("values"), py::arg("start_index") = 0, py::arg("chunk_size") = 1024 * 128, py::arg("bucket_size") = 1024 * 128,
            py::arg("return_values") = false)
        .def("update", &Type::update_with_mask, "add masked values", py::arg("values"), py::arg("masks"), py::arg("start_index") = 0, py::arg("chunk_size") = 1024 * 128,
             py::arg("bucket_size") = 1024 * 128, py::arg("return_values") = false)
        .def("seal", &Type::seal)
        .def("merge", &Type::merge)
        .def("extract", &Type::extract)
        .def("keys", &Type::keys)
        .def("key_array", &Type::key_array)
        .def_property_readonly("count", &Type::count)
        .def("__len__", &Type::length)
        .def("__sizeof__", &Type::bytes_used)
        .def_property_readonly("offset", &Type::offset)
        .def("offsets", &Type::offsets)
        .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
        .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
        .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
        .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; });
}

namespace vaex {
template <class T, class M, template <typename, typename> class Hashmap>
void init_hash_(M m, std::string name, std::string suffix) {
    {
        typedef counter<T, Hashmap> Type;
        std::string countername = "counter_" + name + suffix;
        auto cls = py::class_<Type>(m, countername.c_str())
            .def(py::init<int>())
            // .def("reserve", &Type::reserve)
            .def("counts", &Type::counts)
            ;
        bind_common<Type>(cls);
    }
    {
        std::string ordered_setname = "ordered_set_" + name + suffix;
        typedef ordered_set<T, Hashmap> Type;
        auto cls = py::class_<Type>(m, ordered_setname.c_str())
                       .def(py::init<int, int64_t>(), py::arg("nmaps"), py::arg("limit") = -1)
                       .def(py::init(&Type::create))
                       .def("isin", &Type::isin)
                       .def("flatten_values", &Type::flatten_values)
                       .def("map_ordinal", &Type::map_ordinal)
                       .def_property_readonly("null_value", [](const Type &c) { return c.null_value; })
                       .def_property_readonly("nan_value", [](const Type &c) { return c.nan_value; })
                       .def_readwrite("fingerprint", &Type::fingerprint);
        bind_common<Type>(cls);
    }
    {
        std::string index_hashname = "index_hash_" + name + suffix;
        typedef index_hash<T, Hashmap> Type;
        auto cls = py::class_<Type>(m, index_hashname.c_str())
                       .def(py::init<int>())
                       // .def("reserve", &Type::reserve)
                       .def("map_index", &Type::map_index)
                       .def("map_index", &Type::template map_index_write<int8_t>)
                       .def("map_index", &Type::template map_index_write<int16_t>)
                       .def("map_index", &Type::template map_index_write<int32_t>)
                       .def("map_index", &Type::template map_index_write<int64_t>)
                       .def("map_index_masked", &Type::map_index_with_mask)
                       .def("map_index_masked", &Type::template map_index_with_mask_write<int8_t>)
                       .def("map_index_masked", &Type::template map_index_with_mask_write<int16_t>)
                       .def("map_index_masked", &Type::template map_index_with_mask_write<int32_t>)
                       .def("map_index_masked", &Type::template map_index_with_mask_write<int64_t>)
                       .def("map_index_duplicates", &Type::map_index_duplicates)
                       .def_property_readonly("has_duplicates", [](const Type &c) { return c.has_duplicates; })
            // .def_property_readonly("overflow_size", [](const Type &c) { return c.map.overflow_size(); })
            ;
        bind_common<Type>(cls);
    }
}

} // namespace vaex

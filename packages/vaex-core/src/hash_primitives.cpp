#include "hash_primitives.hpp"


namespace vaex {
template<class T, class M, template<typename, typename> typename Hashmap>
void init_hash_(M m, std::string name, std::string suffix) {
    typedef counter<T, Hashmap> counter_type;
    std::string countername = "counter_" + name + suffix;
    py::class_<counter_type>(m, countername.c_str())
        .def(py::init<>())
        .def("update", &counter_type::update, "add values", py::arg("values"), py::arg("start_index") = 0)
        .def("update", &counter_type::update_with_mask, "add masked values", py::arg("values"), py::arg("masks"), py::arg("start_index") = 0)
        .def("merge", &counter_type::merge)
        .def("extract", &counter_type::extract)
        .def("reserve", &counter_type::reserve)
        .def("keys", &counter_type::keys)
        .def_property_readonly("count", [](const counter_type &c) { return c.count; })
        .def_property_readonly("nan_count", [](const counter_type &c) { return c.nan_count; })
        .def_property_readonly("null_count", [](const counter_type &c) { return c.null_count; })
        .def_property_readonly("has_nan", [](const counter_type &c) { return c.nan_count > 0; })
        .def_property_readonly("has_null", [](const counter_type &c) { return c.null_count > 0; })
    ;
    {
        std::string ordered_setname = "ordered_set_" + name + suffix;
        typedef ordered_set<T, Hashmap> Type;
        py::class_<Type>(m, ordered_setname.c_str())
            .def(py::init<>())
            .def(py::init(&Type::create))
            .def("isin", &Type::isin)
            .def("update", &Type::update, "add values", py::arg("values"), py::arg("start_index") = 0)
            .def("update", &Type::update_with_mask, "add masked values", py::arg("values"), py::arg("masks"), py::arg("start_index") = 0)
            .def("merge", &Type::merge)
            .def("extract", &Type::extract)
            .def("reserve", &Type::reserve)
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
        std::string index_hashname = "index_hash_" + name + suffix;
        typedef index_hash<T, Hashmap> Type;
        py::class_<Type>(m, index_hashname.c_str())
            .def(py::init<>())
            .def("update", &Type::update)
            .def("update", &Type::update_with_mask)
            .def("merge", &Type::merge)
            .def("extract", &Type::extract)
            .def("reserve", &Type::reserve)
            .def("keys", &Type::keys)
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
            .def("__len__", [](const Type &c) { return c.count + (c.null_count > 0) + (c.nan_count > 0); })
            .def_property_readonly("nan_count", [](const Type &c) { return c.nan_count; })
            .def_property_readonly("null_count", [](const Type &c) { return c.null_count; })
            .def_property_readonly("has_nan", [](const Type &c) { return c.nan_count > 0; })
            .def_property_readonly("has_null", [](const Type &c) { return c.null_count > 0; })
            .def_property_readonly("has_duplicates", [](const Type &c) { return c.has_duplicates; })
        ;
    }
}


} // namespace vaex

#include "agg.hpp"
#include "agg_base.hpp"
#include "utils.hpp"

#include <limits>
#include <stdint.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "superstring.hpp"
#include <Python.h>
#include <numpy/arrayobject.h>

using namespace vaex;

template <class Agg, class Base, class Module>
void add_agg(Module m, Base &base, const char *class_name) {
    py::class_<Agg>(m, class_name, base).def(py::init<Grid<> *>(), py::keep_alive<1, 2>()).def_buffer(&agg_buffer_info<Agg>);
}

template <class Agg, class Base, class Module, class A>
void add_agg_arg(Module m, Base &base, const char *class_name) {
    py::class_<Agg>(m, class_name, base).def(py::init<Grid<> *, A>(), py::keep_alive<1, 2>()).def_buffer(&agg_buffer_info<Agg>);
}

namespace vaex {
template <class T, bool FlipEndian>
void add_agg_count_primitive(py::module &m, const py::class_<Aggregator> &base);
void add_agg_count_string(py::module &m, const py::class_<Aggregator> &base);
void add_agg_count_object(py::module &m, const py::class_<Aggregator> &base);

template <class T, bool FlipEndian>
void add_agg_sum_primitive(py::module &m, const py::class_<Aggregator> &base);
template <class T, bool FlipEndian>
void add_agg_sum_moment_primitive(py::module &m, const py::class_<Aggregator> &base);

template <class T, bool FlipEndian>
void add_agg_min_primitive(py::module &m, const py::class_<Aggregator> &base);
template <class T, bool FlipEndian>
void add_agg_max_primitive(py::module &m, const py::class_<Aggregator> &base);

template <class T, bool FlipEndian>
void add_agg_nunique_primitive(py::module &m, const py::class_<Aggregator> &base);

template <class T, bool FlipEndian>
void add_agg_first_primitive(py::module &m, const py::class_<Aggregator> &base);

template <class T, bool FlipEndian>
void add_agg_list_primitive(py::module &m, const py::class_<Aggregator> &base);

void add_agg_nunique_string(py::module &m, py::class_<Aggregator> &base);
void add_agg_list_string(py::module &m, py::class_<Aggregator> &base);
// void add_agg_nunique_primitives(py::module &m, py::class_<Aggregator> &base);
void add_agg_multithreaded(py::module &m, py::class_<Aggregator> &base);
void add_binners(py::module &, py::class_<Binner> &base);

} // namespace vaex

template <class T, class Base, class Module, bool FlipEndian = false>
void add_agg_primitives_(Module m, const Base &base) {
    add_agg_count_primitive<T, FlipEndian>(m, base);

    add_agg_sum_primitive<T, FlipEndian>(m, base);
    add_agg_sum_moment_primitive<T, FlipEndian>(m, base);

    add_agg_min_primitive<T, FlipEndian>(m, base);
    add_agg_max_primitive<T, FlipEndian>(m, base);

    add_agg_nunique_primitive<T, FlipEndian>(m, base);
    add_agg_first_primitive<T, FlipEndian>(m, base);
    add_agg_list_primitive<T, FlipEndian>(m, base);
}

template <class T, class Base, class Module>
void add_agg_primitives(Module m, Base &base) {
    std::string class_name("AggregatorBaseNumpyData");
    class_name += type_name<T>::value;

    // WARNING: Agg is not the super class for AggSum, which does some upcasting, however
    // it does work because the mem layout of the classes is the same
    typedef AggregatorBaseNumpyData<T> Agg;
    py::class_<Agg> aggregator_base(m, class_name.c_str(), py::buffer_protocol(), base);
    aggregator_base.def("__sizeof__", &Agg::bytes_used)
        .def("set_data", &Agg::set_data)
        .def("clear_data_mask", &Agg::clear_data_mask)
        .def("set_data_mask", &Agg::set_data_mask)
        .def_property_readonly("grid", [](const Agg &agg) { return agg.grid; });

    add_agg_primitives_<T, py::class_<Agg>, Module, false>(m, aggregator_base);
    add_agg_primitives_<T, py::class_<Agg>, Module, true>(m, aggregator_base);
}

PYBIND11_MODULE(superagg, m) {
    // m.doc() = "fast statistics/aggregation on grids";
    py::class_<Aggregator> aggregator(m, "Aggregator", py::buffer_protocol());
    { aggregator.def("merge", &Aggregator::merge).def("get_result", &Aggregator::get_result).def("__sizeof__", &Aggregator::bytes_used); }

    py::class_<Binner> binner(m, "Binner");

    {
        typedef Grid<> Type;
        py::class_<Type>(m, "Grid")
            .def(py::init<std::vector<Binner *>>(), py::keep_alive<1, 2>())
            .def("bin", (void (Type::*)(int thread, std::vector<Aggregator *>, size_t)) & Type::bin)
            .def("bin", (void (Type::*)(int thread, std::vector<Aggregator *>)) & Type::bin)
            .def("__len__", [](const Type &grid) { return grid.length1d; })
            .def_property_readonly("binners", [](const Type &grid) { return grid.binners; })
            .def_property_readonly("shapes", [](const Type &grid) { return grid.shapes; })
            .def_property_readonly("strides", [](const Type &grid) { return grid.strides; });
    }

    add_binners(m, binner);
    add_agg_count_string(m, aggregator);
    add_agg_count_object(m, aggregator);
    add_agg_nunique_string(m, aggregator);
    add_agg_list_string(m, aggregator);

#define create(type) add_agg_primitives<type>(m, aggregator);
#include "create_alltypes.hpp"
}

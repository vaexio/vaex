#include "agg_base.hpp"
#include "hash_string.cpp"

namespace vaex {

template <class GridType = counter<>, class IndexType = default_index_type>
class AggNUniqueString : public AggBaseString<GridType> {
  public:
    using Base = AggBaseString<counter<>>;
    using Base::Base;
    using grid_type = GridType;
    AggNUniqueString(Grid<IndexType> *grid, int grids, int threads, bool dropmissing, bool dropnan) : Base(grid, grids, threads), dropmissing(dropmissing), dropnan(dropnan) {}
    void initial_fill(int grid) {}
    virtual py::object get_result() {
        py::array_t<int64_t> result_array(this->grid->length1d);
        auto result = result_array.template mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            for (size_t j = 0; j < this->grid->length1d; j++) {
                result[j] = 0;
                for (int64_t i = 0; i < this->grids; ++i) {
                    counter<> *counter = &this->grid_data[j + i * this->grid->length1d];
                    if (dropmissing) {
                        result[j] += counter->count() - counter->null_count;
                    } else {
                        result[j] += counter->count();
                    }
                }
            }
        }
        auto shape = py::tuple(this->grid->shapes.size());
        for (int i = 0; i < this->grid->shapes.size(); i++) {
            shape[i] = this->grid->shapes[this->grid->shapes.size() - i - 1];
        }
        return result_array.attr("reshape")(shape).attr("T");
    }
    virtual void merge(std::vector<Aggregator *> others) {
        if (others.size() > 0) {
            throw std::runtime_error("merge not implemented");
        }
        // if (grid_data == nullptr)
        //     grid_data = (grid_type *)malloc(sizeof(grid_type) * grid->length1d);
        // for (size_t i = 0; i < this->grid->length1d; i++) {
        //     for (auto j : others) {
        //         auto other = static_cast<AggNUniqueString *>(j);
        //         this->counters[i].merge(other->counters[i]);
        //     }
        //     if (dropmissing)
        //         this->grid_data[i] = counters[i].count() - counters[i].null_count;
        //     else
        //         this->grid_data[i] = counters[i].count();
        // }
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) override {
        auto data_mask_ptr = this->data_mask_ptr[thread];
        auto string_sequence = this->string_sequence[thread];
        auto selection_mask_ptr = this->selection_mask_ptr[thread];
        auto counters = &this->grid_data[grid * this->grid->length1d];
        if (string_sequence == nullptr) {
            throw std::runtime_error("string_sequence not set");
        }
        for (size_t j = 0; j < length; j++) {
            bool masked = false;
            if (selection_mask_ptr && data_mask_ptr[j + offset] == 0)
                continue; // if value is not in selection/filter, don't even consider it
            if (data_mask_ptr && data_mask_ptr[j + offset] == 0)
                masked = true;
            if (string_sequence->is_null(j + offset))
                masked = true;
            if (masked) {
                counters[indices1d[j]].update1_null();
            } else {
                string_view s = string_sequence->view(j + offset);
                counters[indices1d[j]].update1(s);
            }
        }
    }

    bool dropmissing;
    bool dropnan; // not used for strings
};

template <class Agg, class Base, class Module>
void add_agg(Module m, Base &base, const char *class_name) {
    py::class_<Agg>(m, class_name, py::buffer_protocol(), base)
        .def(py::init<Grid<> *, int, int, bool, bool>(), py::keep_alive<1, 2>())
        .def_property_readonly("grid", [](const Agg &agg) { return agg.grid; })
        .def("__sizeof__", &Agg::bytes_used)
        .def("set_data", &Agg::set_data)
        .def("clear_data_mask", &Agg::clear_data_mask)
        .def("set_data_mask", &Agg::set_data_mask)
        .def("set_selection_mask", &Agg::set_selection_mask);
    // .def("merge", &Agg::merge);
}

void add_agg_nunique_string(py::module &m, py::class_<Aggregator> &base) {
    std::string postfix = "string";
    add_agg<AggNUniqueString<>>(m, base, ("AggNUnique_" + postfix).c_str());
}

} // namespace vaex

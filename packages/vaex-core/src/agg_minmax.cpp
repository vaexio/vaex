#include "agg.hpp"
#include "agg_base.hpp"
#include "utils.hpp"

namespace vaex {

template <class DataType = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggMaxPrimitive : public AggregatorPrimitive<DataType, DataType, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, DataType, IndexType>;
    using Base::Base;
    // AggMaxPrimitive(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads) {}
    virtual void initial_fill(int grid) {
        // ignore fill_value
        typedef std::numeric_limits<DataType> limit_type;
        DataType fill_value = limit_type::has_infinity ? -limit_type::infinity() : limit_type::min();
        this->fill(fill_value, grid);
    }
    virtual void merge(std::vector<Aggregator *> others) {
        for (auto i : others) {
            auto other = static_cast<AggMaxPrimitive *>(i);
            for (size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] = std::max(this->grid_data[i], other->grid_data[i]);
            }
        }
    }
    virtual py::object get_result() {
        {
            py::gil_scoped_release release;
            if (!this->grid_used[0]) {
                this->initial_fill(0);
            }
            for (int64_t grid = 1; grid < this->grids; ++grid) {
                if (this->grid_used[grid]) {
                    for (size_t j = 0; j < this->grid->length1d; j++) {
                        this->grid_data[j] = std::max(this->grid_data[j], this->grid_data[j + grid * this->grid->length1d]);
                    }
                }
            }
        }
        py::object numpy = py::module::import("numpy");
        py::object self = py::cast(this);
        return numpy.attr("array")(self).attr("__getitem__")(0);
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) override {
        auto data_mask_ptr = this->data_mask_ptr[thread];
        auto data_ptr = this->data_ptr[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];

        if (data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        if (data_mask_ptr) {
            for (size_t j = 0; j < length; j++) {
                // if not masked
                if (data_mask_ptr[j + offset] == 1) {
                    DataType value = data_ptr[j + offset];
                    if (FlipEndian)
                        value = _to_native(value);
                    if (value != value) // nan
                        continue;
                    grid_data[indices1d[j]] = std::max(value, grid_data[indices1d[j]]);
                }
            }
        } else {
            for (size_t j = 0; j < length; j++) {
                DataType value = data_ptr[offset + j];
                if (FlipEndian)
                    value = _to_native(value);
                if (value == value) // nan check
                    grid_data[indices1d[j]] = std::max(value, grid_data[indices1d[j]]);
            }
        }
    }
};

template <class DataType = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggMinPrimitive : public AggregatorPrimitive<DataType, DataType, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, DataType, IndexType>;
    using Base::Base;
    // AggMinPrimitive(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads) { }
    void initial_fill(int grid) {
        typedef std::numeric_limits<DataType> limit_type;
        DataType fill_value = limit_type::has_infinity ? limit_type::infinity() : limit_type::max();
        this->fill(fill_value, grid);
    }
    virtual void merge(std::vector<Aggregator *> others) {
        for (auto i : others) {
            auto other = static_cast<AggMinPrimitive *>(i);
            for (size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] = std::min(this->grid_data[i], other->grid_data[i]);
            }
        }
    }
    virtual py::object get_result() {
        {
            py::gil_scoped_release release;
            if (!this->grid_used[0]) {
                this->initial_fill(0);
            }
            for (int64_t grid = 1; grid < this->grids; ++grid) {
                if (this->grid_used[grid]) {
                    for (size_t j = 0; j < this->grid->length1d; j++) {
                        this->grid_data[j] = std::min(this->grid_data[j], this->grid_data[j + grid * this->grid->length1d]);
                    }
                }
            }
        }
        py::object numpy = py::module::import("numpy");
        py::object self = py::cast(this);
        return numpy.attr("array")(self).attr("__getitem__")(0);
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) override {
        auto data_mask_ptr = this->data_mask_ptr[thread];
        auto data_ptr = this->data_ptr[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];

        if (data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }

        if (data_mask_ptr) {
            for (size_t j = 0; j < length; j++) {
                // if not masked
                if (data_mask_ptr[j + offset] == 1) {
                    DataType value = data_ptr[j + offset];
                    if (FlipEndian)
                        value = _to_native(value);
                    if (value != value) // nan
                        continue;
                    grid_data[indices1d[j]] = std::min(value, grid_data[indices1d[j]]);
                }
            }
        } else {
            for (size_t j = 0; j < length; j++) {
                DataType value = data_ptr[offset + j];
                if (FlipEndian)
                    value = _to_native(value);
                if (value == value) // nan check
                    grid_data[indices1d[j]] = std::min(value, grid_data[indices1d[j]]);
            }
        }
    }
};

template <class T, bool FlipEndian>
void add_agg_min_primitive(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggMin_");
    class_name += type_name<T>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggMinPrimitive<T, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int>(), py::keep_alive<1, 2>()).def_buffer(&agg_buffer_info<Class>);
}

template <class T, bool FlipEndian>
void add_agg_max_primitive(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggMax_");
    class_name += type_name<T>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggMaxPrimitive<T, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int>(), py::keep_alive<1, 2>()).def_buffer(&agg_buffer_info<Class>);
}

#define create(type)                                                                                                                                                                                   \
    template void add_agg_min_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);                                                                                               \
    template void add_agg_min_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);                                                                                              \
    template void add_agg_max_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);                                                                                               \
    template void add_agg_max_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

} // namespace vaex
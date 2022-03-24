#include "agg.hpp"
#include "agg_base.hpp"
#include "utils.hpp"

namespace vaex {

template <class DataType = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggCountPrimitive : public AggregatorPrimitive<DataType, int64_t, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, int64_t, IndexType>;
    using Base::Base;
    // AggCountPrimitive(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads) { }
    void initial_fill(int grid) { this->fill(0, grid); }

    virtual void merge(std::vector<Aggregator *> others) {
        py::gil_scoped_release release;
        for (auto i : others) {
            auto other = static_cast<AggCountPrimitive *>(i);
            for (size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] += other->grid_data[i];
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
                        this->grid_data[j] += this->grid_data[j + grid * this->grid->length1d];
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
        if (data_mask_ptr || data_ptr) {
            for (size_t j = 0; j < length; j++) {
                // if not masked
                if (data_mask_ptr == nullptr || data_mask_ptr[j + offset] == 1) {
                    // and not nan (TODO: we can skip this for non-floats)
                    if (data_ptr) {
                        DataType value = data_ptr[j + offset];
                        if (FlipEndian)
                            value = _to_native(value);
                        if (value != value) // nan
                            continue;
                    }
                    grid_data[indices1d[j]] += 1;
                }
            }
        } else {
            for (size_t j = 0; j < length; j++) {
                grid_data[indices1d[j]] += 1;
            }
        }
    }
};

template <class GridType = uint64_t, class IndexType = default_index_type>
class AggCountObject : public AggBaseObject<GridType, IndexType> {
  public:
    using Base = AggBaseObject<GridType, IndexType>;
    using Type = AggCountObject<GridType, IndexType>;
    using Base::Base;
    // AggCountObject(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads) { }
    void initial_fill(int grid) { this->fill(0, grid); }
    virtual void merge(std::vector<Aggregator *> others) {
        for (auto i : others) {
            auto other = static_cast<AggCountObject *>(i);
            for (size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] += other->grid_data[i];
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
                        this->grid_data[j] += this->grid_data[j + grid * this->grid->length1d];
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
        auto objects = this->objects[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];
        if (objects == nullptr) {
            throw std::runtime_error("object data not set");
        }
        if (data_mask_ptr == nullptr) {
            for (size_t j = 0; j < length; j++) {
                PyObject *obj = objects[j + offset];
                bool none = (obj == Py_None);
                bool _isnan = PyFloat_Check(obj) && std::isnan(PyFloat_AsDouble(obj));
                grid_data[indices1d[j]] += (none || _isnan ? 0 : 1);
            }
        } else {
            for (size_t j = 0; j < length; j++) {
                PyObject *obj = objects[j + offset];
                bool none = (obj == Py_None);
                bool _isnan = PyFloat_Check(obj) && std::isnan(PyFloat_AsDouble(obj));
                bool masked = data_mask_ptr[j + offset] == 0;
                grid_data[indices1d[j]] += (none || masked || _isnan ? 0 : 1);
            }
        }
    }
    virtual bool can_release_gil() { return false; };
};

template <class GridType = uint64_t, class IndexType = default_index_type>
class AggCountString : public AggBaseString<GridType, IndexType> {
  public:
    using Base = AggBaseString<GridType, IndexType>;
    using Type = AggCountString<GridType, IndexType>;
    using Base::Base;
    // AggCountString(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads) { }
    void initial_fill(int grid) { this->fill(0, grid); }
    virtual void merge(std::vector<Aggregator *> others) {
        for (auto j : others) {
            auto other = static_cast<AggCountString *>(j);
            for (size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] += other->grid_data[i];
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
                        this->grid_data[j] += this->grid_data[j + grid * this->grid->length1d];
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
        auto string_sequence = this->string_sequence[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];
        if (string_sequence == nullptr) {
            throw std::runtime_error("string_sequence not set");
        }
        if (!string_sequence->has_null() && data_mask_ptr == nullptr) {
            // fast path
            for (size_t j = 0; j < length; j++) {
                grid_data[indices1d[j]] += 1;
            }
        } else if (string_sequence->has_null() && data_mask_ptr == nullptr) {
            for (size_t j = 0; j < length; j++) {
                grid_data[indices1d[j]] += string_sequence->is_null(j + offset) ? 0 : 1;
            }
        } else if (!string_sequence->has_null() && data_mask_ptr != nullptr) {
            for (size_t j = 0; j < length; j++) {
                bool masked = data_mask_ptr[j + offset] == 0;
                grid_data[indices1d[j]] += masked ? 0 : 1;
            }
        } else if (string_sequence->has_null() && data_mask_ptr != nullptr) {
            for (size_t j = 0; j < length; j++) {
                bool masked = data_mask_ptr[j + offset] == 0;
                grid_data[indices1d[j]] += string_sequence->is_null(j + offset) || masked ? 0 : 1;
            }
        }
    }
};

template <class T, bool FlipEndian>
void add_agg_count_primitive(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggCount_");
    class_name += type_name<T>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggCountPrimitive<T, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int>(), py::keep_alive<1, 2>()).def_buffer(&agg_buffer_info<Class>);
}

void add_agg_count_string(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggCount_string");
    using Class = AggCountString<>;
    add_agg_binding_1arg<Class>(m, base, class_name.c_str());
}

void add_agg_count_object(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggCount_object");
    using Class = AggCountObject<>;
    add_agg_binding_1arg<Class>(m, base, class_name.c_str());
}

#define create(type)                                                                                                                                                                                   \
    template void add_agg_count_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);                                                                                             \
    template void add_agg_count_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

} // namespace vaex

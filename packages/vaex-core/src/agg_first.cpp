#include "agg_base.hpp"
#include "utils.hpp"

namespace vaex {

template <class DataType = double, class DataType2 = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggFirstPrimitive : public AggregatorPrimitive<DataType, DataType, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, DataType, IndexType>;
    using Base::Base;
    using data_type = DataType;
    using data_type2 = DataType2;

    AggFirstPrimitive(Grid<IndexType> *grid, int grids, int threads, bool invert)
        : Base(grid, grids, threads), data_ptr2(threads), data_size2(threads), invert(invert) {
        grid_data_order = new data_type2[this->count()];
        cell_masked = new bool[this->count()];
    }
    void initial_fill(int grid) {
        this->fill(99, grid);
        typedef std::numeric_limits<data_type2> limit_type;
        std::fill(grid_data_order + this->grid->length1d * grid, grid_data_order + this->grid->length1d * (grid + 1), invert ? limit_type::min() : limit_type::max());
        // 1 is masked
        std::fill(cell_masked + this->grid->length1d * grid, cell_masked + this->grid->length1d * (grid + 1), 1);
    }
    virtual ~AggFirstPrimitive() { delete[] grid_data_order; }
    void set_data(int thread, py::buffer ar, size_t index) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if (index == 1) {
            this->data_ptr2[thread] = (DataType2 *)info.ptr;
            this->data_size2[thread] = info.shape[0];
        } else {
            this->data_ptr[thread] = (DataType *)info.ptr;
            this->data_size[thread] = info.shape[0];
        }
    }
    virtual void merge(std::vector<Aggregator *> others) {
        const bool invert = this->invert;
        // for (auto i : others) {
        //     auto other = static_cast<AggFirstPrimitive *>(i);
        //     for (size_t i = 0; i < this->grid->length1d; i++) {
        //         if (invert) {
        //             if (other->grid_data_order[i] > this->grid_data_order[i]) {
        //                 this->grid_data[i] = other->grid_data[i];
        //                 this->grid_data_order[i] = other->grid_data_order[i];
        //             }
        //         } else {
        //             if (other->grid_data_order[i] < this->grid_data_order[i]) {
        //                 this->grid_data[i] = other->grid_data[i];
        //                 this->grid_data_order[i] = other->grid_data_order[i];
        //             }
        //         }
        //     }
        // }
    }
    virtual pybind11::object get_result() {
        const bool invert = this->invert;
        py::array_t<bool> mask(this->grid->length1d);
        {
            py::gil_scoped_release release;
            if (!this->grid_used[0]) {
                this->initial_fill(0);
            }
            for (int64_t grid = 1; grid < this->grids; ++grid) {
                if (this->grid_used[grid]) {
                    for (size_t j = 0; j < this->grid->length1d; j++) {
                        int64_t j2 = j + grid * this->grid->length1d;
                        if(cell_masked[j2] == 1) {
                            // if j2 is masked, we can skip it
                        } else {
                            if(cell_masked[j] == 1) {
                                // if j is masked, we can just assign
                                this->grid_data[j] = this->grid_data[j2];
                                grid_data_order[j] = grid_data_order[j2];
                                cell_masked[j] = 0;
                            } else {
                                // if both unmasked, we need to compare
                                if (invert) {
                                    if (grid_data_order[j2] > grid_data_order[j]) {
                                        this->grid_data[j] = this->grid_data[j2];
                                        grid_data_order[j] = grid_data_order[j2];
                                    }
                                } else {
                                    if (grid_data_order[j2] < grid_data_order[j]) {
                                        this->grid_data[j] = this->grid_data[j2];
                                        grid_data_order[j] = grid_data_order[j2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            bool *mask_ptr = mask.mutable_data(0);
            for (size_t j = 0; j < this->grid->length1d; j++) {
                mask_ptr[j] = cell_masked[j];
            }
        }
        py::object numpy = py::module::import("numpy");
        py::object numpy_ma = py::module::import("numpy.ma");
        py::object self = py::cast(this);
        py::object data = numpy.attr("array")(self).attr("__getitem__")(0);
        using namespace pybind11::literals; // to bring in the `_a` literal
        auto shape = py::tuple(this->grid->shapes.size());
        for (int i = 0; i < this->grid->shapes.size(); i++) {
            shape[i] = this->grid->shapes[this->grid->shapes.size() - i - 1];
        }
        return numpy_ma.attr("array")(data, "mask"_a=mask.attr("reshape")(shape).attr("T"));
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) {
        auto data_ptr = this->data_ptr[thread];
        auto data_ptr2 = this->data_ptr2[thread];
        auto data_mask_ptr = this->data_mask_ptr[thread];
        // auto data_mask_ptr2 = this->data_mask_ptr2[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];
        auto grid_data_order = &this->grid_data_order[grid * this->grid->length1d];
        auto cell_masked = &this->cell_masked[grid * this->grid->length1d];

        if (data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        // if (data_ptr2 == nullptr) {
        //     throw std::runtime_error("data2 not set");
        // }
        const bool invert = this->invert;
        for (size_t j = 0; j < length; j++) {
            if((data_mask_ptr == nullptr) || (data_mask_ptr[j]) == 1) {
                DataType value = data_ptr[offset + j];
                DataType2 value_order = data_ptr2 == nullptr ? offset + j : data_ptr2[offset + j];
                if (FlipEndian) {
                    value = _to_native(value);
                    value_order = _to_native(value_order);
                }
                if (value == value && value_order == value_order) { // nan check
                    IndexType i = indices1d[j];
                    if(cell_masked[i] == 1) {
                        // if masked, we directly assign
                        grid_data[i] = value;
                        cell_masked[i] = 0;
                        grid_data_order[i] = value_order;
                    } else {
                        // otherwise we need to compare
                        if (invert) {
                            if (value_order > grid_data_order[i]) {
                                grid_data[i] = value;
                                cell_masked[i] = 0;
                                grid_data_order[i] = value_order;
                            }
                        } else {
                            if (value_order < grid_data_order[i]) {
                                grid_data[i] = value;
                                cell_masked[i] = 0;
                                grid_data_order[i] = value_order;
                            }
                        }
                    }
                }
            }
        }
    }
    data_type2 *grid_data_order;
    bool *cell_masked ;

    std::vector<data_type2 *> data_ptr2;
    std::vector<uint64_t> data_size2;
    std::vector<uint8_t *> data_mask_ptr2;
    std::vector<uint64_t> data_mask_size2;
    bool invert; // intead of creating 2x as many templates
};



template <class T, class T2, bool FlipEndian>
void add_agg_first_primitive_mixed(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggFirst_");
    class_name += type_name<T>::value;
    class_name += "_";
    class_name += type_name<T2>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggFirstPrimitive<T, T2, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int, bool>(), py::keep_alive<1, 2>()).def_buffer(&agg_buffer_info<Class>); // .def("set_data_mask2", &Class::set_data_mask2)
    // .def("clear_data_mask2", &Class::clear_data_mask2);
}

template <class T, bool FlipEndian>
void add_agg_first_primitive(py::module &m, const py::class_<Aggregator> &base) {
    // add_agg_first_primitive_mixed<std::string, type, FlipEndian>(m, base);
#define create(type) add_agg_first_primitive_mixed<T, type, FlipEndian>(m, base);
#include "create_alltypes.hpp"
}

#undef create
#define create(type)                                                                                                                                                                                   \
    template void add_agg_first_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);                                                                                             \
    template void add_agg_first_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

} // namespace vaex
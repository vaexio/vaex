#include "agg.hpp"

namespace vaex {

template <class DataType = double, class DataType2 = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggFirst : public AggregatorPrimitive<DataType, DataType, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, DataType, IndexType>;
    using Base::Base;
    AggFirst(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads), data_ptr2(threads), data_size2(threads), data_mask_ptr2(threads), data_mask_size2(threads) {
        grid_data_order = (DataType *)malloc(sizeof(DataType) * grid->length1d);
        typedef std::numeric_limits<DataType> limit_type;
        std::fill(grid_data_order, grid_data_order + grid->length1d, limit_type::max());
    }
    virtual ~AggFirst() { free(grid_data_order); }
    void set_data(int threads, py::buffer ar, size_t index) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if (index == 1) {
            this->data_ptr2 = (DataType *)info.ptr;
            this->data_size2 = info.shape[0];
        } else {
            this->data_ptr = (DataType *)info.ptr;
            this->data_size = info.shape[0];
        }
    }
    void set_data_mask2(int threads, py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr2 = (uint8_t *)info.ptr;
        this->data_mask_size2 = info.shape[0];
    }
    virtual void merge(std::vector<Aggregator *> others) {
        for (auto i : others) {
            auto other = static_cast<AggFirst *>(i);
            for (size_t i = 0; i < this->grid->length1d; i++) {
                if (other->grid_data_order[i] < this->grid_data_order[i]) {
                    this->grid_data[i] = other->grid_data[i];
                    this->grid_data_order[i] = other->grid_data_order[i];
                }
            }
        }
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) {
        auto data_ptr = this->data_ptr[thread];
        auto data_ptr2 = this->data_ptr[thread];
        auto data_mask_ptr = this->data_mask_ptr[thread];
        auto data_mask_ptr2 = this->data_mask_ptr2[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];

        if (data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        if (data_ptr2 == nullptr) {
            throw std::runtime_error("data2 not set");
        }
        // TODO: masked support
        for (size_t j = 0; j < length; j++) {
            DataType value = data_ptr[offset + j];
            DataType value_order = data_ptr2[offset + j];
            if (FlipEndian) {
                value = _to_native(value);
                value_order = _to_native(value_order);
            }
            if (value == value && value_order == value_order) { // nan check
                IndexType i = indices1d[j];
                if (value_order < grid_data_order[i]) {
                    grid_data[i] = value;
                    grid_data_order[i] = value_order;
                }
            }
        }
    }
    DataType2 *grid_data_order;

    std::vector<data_type *> data_ptr2;
    std::vector<uint64_t> data_size2;
    std::vector<uint8_t *> data_mask_ptr2;
    std::vector<uint64_t> data_mask_size2;

    uint64_t data_size2;
    uint8_t *data_mask_ptr2;
    uint64_t data_mask_size2;
};

template <class T, bool FlipEndian>
void add_agg_first_primitive(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggFirst_");
    class_name += type_name<T>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggFirstPrimitive<T, T, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int>(), py::keep_alive<1, 2>()).def_buffer(&Class::buffer_info);
}

// TODO: implement string
// void add_agg_first_string(py::module &m, const py::class_<Aggregator> &base) {
//     std::string class_name = std::string("AggCount_string");
//     using Class = AggCountString<>;
//     add_agg_binding_1arg<Class>(m, base, class_name.c_str());
// }

#define create(type)                                                                                                                                                                                   \
    template void add_agg_first_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);                                                                                             \
    template void add_agg_first_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

} // namespace vaex
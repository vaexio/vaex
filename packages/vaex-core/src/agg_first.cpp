#include "agg.hpp"

namespace vaex {

template <class DataType = double, class DataType2 = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggFirst : public AggregatorPrimitive<DataType, DataType, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, DataType, IndexType>;
    using Base::Base;
    AggFirst(Grid<IndexType> *grid) : Base(grid) {
        grid_data_order = (DataType *)malloc(sizeof(DataType) * grid->length1d);
        typedef std::numeric_limits<DataType> limit_type;
        std::fill(grid_data_order, grid_data_order + grid->length1d, limit_type::max());
    }
    virtual ~AggFirst() { free(grid_data_order); }
    void set_data(py::buffer ar, size_t index) {
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
    void set_data_mask2(py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr2 = (uint8_t *)info.ptr;
        this->data_mask_size2 = info.shape[0];
    }
    virtual void reduce(std::vector<Aggregator *> others) {
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
    virtual void aggregate(default_index_type *indices1d, size_t length, uint64_t offset) {
        if (this->data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        if (this->data_ptr2 == nullptr) {
            throw std::runtime_error("data2 not set");
        }
        // TODO: masked support
        for (size_t j = 0; j < length; j++) {
            DataType value = this->data_ptr[offset + j];
            DataType value_order = this->data_ptr2[offset + j];
            if (FlipEndian) {
                value = _to_native(value);
                value_order = _to_native(value_order);
            }
            if (value == value && value_order == value_order) { // nan check
                IndexType i = indices1d[j];
                if (value_order < grid_data_order[i]) {
                    this->grid_data[i] = value;
                    this->grid_data_order[i] = value_order;
                }
            }
        }
    }
    DataType *grid_data_order;
    DataType *data_ptr2;
    uint64_t data_size2;
    uint8_t *data_mask_ptr2;
    uint64_t data_mask_size2;
};
} // namespace vaex
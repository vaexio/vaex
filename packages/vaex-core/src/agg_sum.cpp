#include "agg_base.hpp"
#include "utils.hpp"

namespace vaex {

template <class T>
struct upcast {};

template <>
struct upcast<float> {
    typedef double type;
};

template <>
struct upcast<double> {
    typedef double type;
};

template <>
struct upcast<bool> {
    typedef int64_t type;
};

template <>
struct upcast<int8_t> {
    typedef int64_t type;
};

template <>
struct upcast<int16_t> {
    typedef int64_t type;
};

template <>
struct upcast<int32_t> {
    typedef int64_t type;
};

template <>
struct upcast<int64_t> {
    typedef int64_t type;
};

template <>
struct upcast<uint8_t> {
    typedef uint64_t type;
};

template <>
struct upcast<uint16_t> {
    typedef uint64_t type;
};

template <>
struct upcast<uint32_t> {
    typedef uint64_t type;
};

template <>
struct upcast<uint64_t> {
    typedef uint64_t type;
};

// see agg_sum or agg_count for example usage
template <class Derived, class DataType = double, class GridType = DataType, class IndexType = default_index_type, bool FlipEndian = false>
class AggregatorPrimitiveCRTP : public AggregatorPrimitive<DataType, GridType, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, GridType, IndexType>;
    using grid_type = GridType;
    using Base::Base;

    virtual void merge(std::vector<Aggregator *> others) {
        for (auto i : others) {
            auto other = static_cast<Derived *>(i);
            for (size_t i = 0; i < this->grid->length1d; i++) {
                static_cast<const Derived &>(*this).op_reduce_mutate(this->grid_data[i], other->grid_data[i]);
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
                        this->grid_data[j] = static_cast<const Derived &>(*this).op_reduce(this->grid_data[j], this->grid_data[j + grid * this->grid->length1d]);
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
        grid_type *grid_data = &this->grid_data[grid * this->grid->length1d];
        if ((data_ptr == nullptr) && (requires_arg(0))) {
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
                    static_cast<const Derived &>(*this).op_mutate(grid_data[indices1d[j]], value);
                }
            }
        } else {
            for (size_t j = 0; j < length; j++) {
                DataType value = data_ptr[offset + j];
                if (FlipEndian)
                    value = _to_native(value);
                if (value == value) { // nan check {
                    static_cast<const Derived &>(*this).op_mutate(grid_data[indices1d[j]], value);
                }
            }
        }
    }
    virtual bool requires_arg(int i) = 0;
};

template <class DataType = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggSumPrimitive : public AggregatorPrimitiveCRTP<AggSumPrimitive<DataType, IndexType, FlipEndian>, DataType, typename upcast<DataType>::type, IndexType, FlipEndian> {
  public:
    using Base = AggregatorPrimitiveCRTP<AggSumPrimitive<DataType, IndexType, FlipEndian>, DataType, typename upcast<DataType>::type, IndexType, FlipEndian>;
    using Base::Base;
    using grid_type = typename Base::grid_type;

    // AggSumPrimitive(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads) { }
    void initial_fill(int grid) { this->fill(0, grid); }

    void op_mutate(grid_type &a, grid_type b) const { a += b; }
    grid_type op(grid_type a, grid_type b) const { return a + b; }

    void op_reduce_mutate(grid_type &a, grid_type b) const { a += b; }
    grid_type op_reduce(grid_type a, grid_type b) const { return a + b; }
    virtual bool requires_arg(int i) { return true; }
};

template <class DataType = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggSumMomentPrimitive : public AggregatorPrimitiveCRTP<AggSumMomentPrimitive<DataType, IndexType, FlipEndian>, DataType, typename upcast<DataType>::type, IndexType, FlipEndian> {
  public:
    using Base = AggregatorPrimitiveCRTP<AggSumMomentPrimitive<DataType, IndexType, FlipEndian>, DataType, typename upcast<DataType>::type, IndexType, FlipEndian>;
    using Base::Base;
    using grid_type = typename Base::grid_type;

    AggSumMomentPrimitive(Grid<IndexType> *grid, int grids, int threads, uint32_t moment) : Base(grid, grids, threads), moment(moment) {}
    void initial_fill(int grid) { this->fill(0, grid); }

    void op_mutate(grid_type &a, grid_type b) const { a += pow(b, moment); }
    grid_type op(grid_type a, grid_type b) const { return a + pow(b, moment); }

    void op_reduce_mutate(grid_type &a, grid_type b) const { a += b; }
    grid_type op_reduce(grid_type a, grid_type b) const { return a + b; }
    virtual bool requires_arg(int i) { return true; }
    uint32_t moment;
};

// template <class StorageType = double, class IndexType = default_index_type, bool FlipEndian = false>
// class AggSumMoment : public AggregatorPrimitive<StorageType, typename upcast<StorageType>::type, IndexType> {
//   public:
//     using Base = AggregatorPrimitive<StorageType, typename upcast<StorageType>::type, IndexType>;
//     using Base::Base;
//     AggSumMoment(Grid<IndexType> *grid, uint32_t moment) : Base(grid), moment(moment) {}
//     virtual void merge(std::vector<Aggregator *> others) {
//         for (auto i : others) {
//             auto other = static_cast<AggSumMoment *>(i);
//             for (size_t i = 0; i < this->grid->length1d; i++) {
//                 this->grid_data[i] = this->grid_data[i] + other->grid_data[i];
//             }
//         }
//     }
//     virtual void aggregate(default_index_type *indices1d, size_t length, uint64_t offset) {
//         if (this->data_ptr == nullptr) {
//             throw std::runtime_error("data not set");
//         }

//         if (this->data_mask_ptr) {
//             for (size_t j = 0; j < length; j++) {
//                 // if not masked
//                 if (this->data_mask_ptr[j + offset] == 1) {
//                     typename Base::grid_type value = this->data_ptr[j + offset];
//                     if (FlipEndian)
//                         value = _to_native(value);
//                     if (value != value) // nan
//                         continue;
//                     this->grid_data[indices1d[j]] += pow(value, moment);
//                 }
//             }
//         } else {
//             for (size_t j = 0; j < length; j++) {
//                 typename Base::grid_type value = this->data_ptr[offset + j];
//                 if (FlipEndian)
//                     value = _to_native(value);
//                 if (value == value) // nan check
//                     this->grid_data[indices1d[j]] += pow(value, moment);
//             }
//         }
//     }
//     size_t moment;
// };

template <class T, bool FlipEndian>
void add_agg_sum_primitive(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggSum_");
    class_name += type_name<T>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggSumPrimitive<T, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int>(), py::keep_alive<1, 2>()).def_buffer(&agg_buffer_info<Class>);
}

template <class T, bool FlipEndian>
void add_agg_sum_moment_primitive(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggSumMoment_");
    class_name += type_name<T>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggSumMomentPrimitive<T, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int, uint32_t>(), py::keep_alive<1, 2>()).def_buffer(&agg_buffer_info<Class>);
}

#define create(type)                                                                                                                                                                                   \
    template void add_agg_sum_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);                                                                                              \
    template void add_agg_sum_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

#undef create
#define create(type)                                                                                                                                                                                   \
    template void add_agg_sum_moment_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);                                                                                       \
    template void add_agg_sum_moment_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

} // namespace vaex

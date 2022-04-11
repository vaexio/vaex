#include "agg_base.hpp"
#include "hash_primitives.hpp"
#include "utils.hpp"

namespace vaex {

template <class DataType = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggNUniquePrimitive : public AggregatorPrimitive<DataType, counter<DataType, hashmap_primitive>, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, counter<DataType, hashmap_primitive>, IndexType>;
    using Base::Base;
    using grid_type = counter<DataType, hashmap_primitive>;
    using data_type = DataType;
    AggNUniquePrimitive(Grid<IndexType> *grid, int grids, int threads, bool dropmissing, bool dropnan) : Base(grid, grids, threads), dropmissing(dropmissing), dropnan(dropnan) {}
    void initial_fill(int grid) {}
    virtual py::object get_result() {
        py::array_t<int64_t> result_array(this->grid->length1d);
        auto result = result_array.template mutable_unchecked<1>();
        {
            if (this->grids != 1) {
                throw std::runtime_error("Expected 1 grid");
            }
            py::gil_scoped_release release;
            for (size_t j = 0; j < this->grid->length1d; j++) {
                result[j] = 0;
                for (int64_t i = 0; i < this->grids; ++i) {
                    grid_type *counter = &this->grid_data[j + i * this->grid->length1d];
                    auto count = counter->count();
                    if (dropmissing)
                        count -= counter->null_count;
                    if (dropnan)
                        count -= counter->nan_count;
                    result[j] += count;
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
        //         auto other = static_cast<AggNUniquePrimitive *>(j);
        //         this->counters[i].merge(other->counters[i]);
        //     }
        //     if (dropmissing)
        //         this->grid_data[i] = counters[i].count() - counters[i].null_count;
        //     else
        //         this->grid_data[i] = counters[i].count();
        // }
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) override {
        auto data_ptr = this->data_ptr[thread];
        auto data_mask_ptr = this->data_mask_ptr[thread];
        auto selection_mask_ptr = this->selection_mask_ptr[thread];
        auto counters = &this->grid_data[grid * this->grid->length1d];
        if (data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        for (size_t j = 0; j < length; j++) {
            bool masked = false;
            if (selection_mask_ptr && data_mask_ptr[j + offset] == 0)
                continue; // if value is not in selection/filter, don't even consider it
            if (data_mask_ptr && data_mask_ptr[j + offset] == 0)
                masked = true;
            if (masked) {
                counters[indices1d[j]].update1_null();
            } else {
                data_type value = data_ptr[j + offset];
                if (FlipEndian)
                    value = _to_native(value);
                if (value != value) // nan
                    counters[indices1d[j]].update1_nan();
                else
                    counters[indices1d[j]].update1(value);
            }
        }
    }

    bool dropmissing;
    bool dropnan; // not used for strings
};

// template <, class IndexType = default_index_type, bool FlipEndian = false>
//     using Base = AggregatorPrimitive<DataType, counter<DataType, hashmap_primitive>, IndexType>;
//     using Base::Base;
//     using data_type = DataType;
//     AggNUniqueString(Grid<IndexType> *grid, int grids, int threads, bool dropmissing, bool dropnan) : Base(grid, grids, threads), dropmissing(dropmissing), dropnan(dropnan) {
//     virtual void merge(std::vector<Aggregator *> others) {}
//     virtual py::object get_result() override { return py::none(); }
//     virtual void aggregate(int thread, default_index_type *indices1d, size_t length, uint64_t offset) override {
//         auto data_mask_ptr = this->data_mask_ptr[thread];
//         auto data_ptr = this->data_ptr[thread];
//         auto counters = &this->grid_data[thread * this->grid->length1d];
//         if (data_ptr == nullptr) {
//             throw std::runtime_error("data not set");
//         }
//         for (size_t j = 0; j < length; j++) {
//             bool masked = false;
//             // if (selection_mask_ptr && data_mask_ptr[j + offset] == 0)
//             //     continue; // if value is not in selection/filter, don't even consider it
//             if (data_mask_ptr && data_mask_ptr[j + offset] == 0)
//                 masked = true;
//             if (masked) {
//                 counters[indices1d[j]].update1_null();
//             } else {
//                 data_type value = data_ptr[j + offset];
//                 if (FlipEndian)
//                     value = _to_native(value);
//                 if (value != value) // nan
//                     counters[indices1d[j]].update1_nan();
//                 else
//                     counters[indices1d[j]].update1(value);
//             }
//         }
//     }
// };

// template <class DataType = double, class GridType = uint64_t, class IndexType = default_index_type, bool FlipEndian = false>
// class AggNUnique : public Aggregator {
//   public:
//     using Type = AggNUnique<DataType, GridType, IndexType, FlipEndian>;
//     using Counter = counter<DataType, hashmap_primitive>; // TODO: do we want a prime growth variant?
//     using index_type = IndexType;
//     using grid_type = GridType;
//     using data_type = DataType;
//     AggNUnique(Grid<IndexType> *grid, bool dropmissing, bool dropnan)
//         : grid(grid), grid_data(nullptr), data_ptr(nullptr), data_mask_ptr(nullptr), selection_mask_ptr(nullptr), dropmissing(dropmissing), dropnan(dropnan) {
//         counters = new Counter[grid->length1d];
//     }
//     virtual ~AggNUnique() {
//         if (grid_data)
//             free(grid_data);
//         delete[] counters;
//     }
//     virtual size_t bytes_used() { return sizeof(grid_type) * grid->length1d; }
//     virtual void reduce(std::vector<Aggregator *> others) {
//         if (grid_data == nullptr) {
//             grid_data = (grid_type *)malloc(sizeof(grid_type) * grid->length1d);
//         }
//         for (size_t i = 0; i < this->grid->length1d; i++) {
//             for (auto j : others) {
//                 auto other = static_cast<AggNUnique *>(j);
//                 this->counters[i].merge(other->counters[i]);
//             }
//             grid_data[i] = counters[i].count();
//             if (dropmissing)
//                 grid_data[i] -= counters[i].null_count;
//             if (dropnan)
//                 grid_data[i] -= counters[i].nan_count;
//         }
//     }
//     virtual void aggregate(default_index_type *indices1d, size_t length, uint64_t offset) {
//         if (this->data_ptr == nullptr) {
//             throw std::runtime_error("data not set");
//         }
//         for (size_t j = 0; j < length; j++) {
//             bool masked = false;
//             if (this->selection_mask_ptr && this->data_mask_ptr[j + offset] == 0)
//                 continue; // if value is not in selection/filter, don't even consider it
//             if (this->data_mask_ptr && this->data_mask_ptr[j + offset] == 0)
//                 masked = true;
//             if (masked) {
//                 this->counters[indices1d[j]].update1_null();
//             } else {
//                 data_type value = this->data_ptr[j + offset];
//                 if (FlipEndian)
//                     value = _to_native(value);
//                 if (value != value) // nan
//                     this->counters[indices1d[j]].update1_nan();
//                 else
//                     this->counters[indices1d[j]].update1(value);
//             }
//         }
//     }
//     void set_data(py::buffer ar, size_t index) {
//         py::buffer_info info = ar.request();
//         if (info.ndim != 1) {
//             throw std::runtime_error("Expected a 1d array");
//         }
//         this->data_ptr = (data_type *)info.ptr;
//         this->data_size = info.shape[0];
//     }
//     void clear_data_mask() {
//         this->data_mask_ptr = nullptr;
//         this->data_mask_size = 0;
//     }
//     void set_data_mask(py::buffer ar) {
//         py::buffer_info info = ar.request();
//         if (info.ndim != 1) {
//             throw std::runtime_error("Expected a 1d array");
//         }
//         this->data_mask_ptr = (uint8_t *)info.ptr;
//         this->data_mask_size = info.shape[0];
//     }
//     void set_selection_mask(py::buffer ar) {
//         py::buffer_info info = ar.request();
//         if (info.ndim != 1) {
//             throw std::runtime_error("Expected a 1d array");
//         }
//         this->selection_mask_ptr = (uint8_t *)info.ptr;
//         this->selection_mask_size = info.shape[0];
//     }
//     Grid<IndexType> *grid;
//     grid_type *grid_data;
//     Counter *counters;
//     data_type *data_ptr;
//     uint64_t data_size;
//     uint8_t *data_mask_ptr;
//     uint64_t data_mask_size;
//     uint8_t *selection_mask_ptr;
//     uint64_t selection_mask_size;
//     bool dropmissing;
//     bool dropnan;
// };

template <class T, bool FlipEndian = false>
void add_agg_nunique_primitive(py::module &m, const py::class_<Aggregator> &base) {
    // add_agg<AggCount<T, default_index_type, FlipEndian>, Base, Module>(m, base, ("AggCount_" + postfix).c_str());
    // add_agg<AggNUnique<T, uint64_t, default_index_type, FlipEndian>>(m, base, ("AggNUnique_" + postfix).c_str());
    std::string class_name = std::string("AggNUnique_");
    class_name += type_name<T>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggNUniquePrimitive<T, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int, bool, bool>(), py::keep_alive<1, 2>());
}

#define create(type)                                                                                                                                                                                   \
    template void add_agg_nunique_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);                                                                                           \
    template void add_agg_nunique_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

} // namespace vaex
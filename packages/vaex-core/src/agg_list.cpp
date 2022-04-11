#include "agg_base.hpp"
#include "utils.hpp"

namespace vaex {

template <class DataType = double, class DataType2 = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggListPrimitive : public AggregatorPrimitive<DataType, std::vector<typename FixBool<DataType>::value>, IndexType> {
  public:
    using Base = AggregatorPrimitive<DataType, std::vector<typename FixBool<DataType>::value>, IndexType>;
    using Base::Base;
    using data_type_fixed = typename FixBool<DataType>::value;
    using data_type = DataType;
    using data_type2 = DataType2;
    using data_type2_fixed = typename FixBool<DataType2>::value;

    AggListPrimitive(Grid<IndexType> *grid, int grids, int threads, bool dropnan, bool dropnull)
        : Base(grid, grids, threads), data_ptr2(threads), data_size2(threads), dropnan(dropnan), dropnull(dropnull) {
        if (grids != 1) {
            throw std::runtime_error("list aggregation only accepts 1 grid");
        }
        nan_count = new int64_t[this->count()];
        null_count = new int64_t[this->count()];
    }
    void initial_fill(int grid) {
        std::fill(nan_count + this->grid->length1d * grid, nan_count + this->grid->length1d * (grid + 1), 0);
        std::fill(null_count + this->grid->length1d * grid, null_count + this->grid->length1d * (grid + 1), 0);
    }
    virtual ~AggListPrimitive() {
        delete[] nan_count;
        delete[] null_count;
    }
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
    virtual void merge(std::vector<Aggregator *> others) {}
    virtual pybind11::object get_result() {
        py::array_t<data_type_fixed> values_array;
        py::array_t<int64_t> offsets_array;
        {
            if (!this->grid_used[0]) {
                this->initial_fill(0);
            }
            // we only have a single grid
            auto grid_data = &this->grid_data[0];
            offsets_array = py::array_t<int64_t>(this->grid->length1d + 1);
            int64_t *offsets = offsets_array.mutable_data(0);
            int64_t offset = 0;
            offsets[0] = 0;
            // first we fill the offsets
            for (size_t j = 0; j < this->grid->length1d; j++) {
                offset += grid_data[j].size();
                offset += this->nan_count[j];
                offset += this->null_count[j];
                offsets[j + 1] = offset;
            }

            int64_t flat_length = offset;
            values_array = py::array_t<data_type_fixed>(flat_length);
            data_type_fixed *values = values_array.mutable_data(0);
            py::gil_scoped_release release;
            for (size_t j = 0; j < this->grid->length1d; j++) {
                std::copy(grid_data[j].begin(), grid_data[j].end(), values + offsets[j]);
                std::fill(values + offsets[j] + grid_data[j].size(), values + offsets[j] + grid_data[j].size() + this->nan_count[j], std::numeric_limits<data_type_fixed>::quiet_NaN());
            }
        }
        py::object vaex_arrow_convert = py::module::import("vaex.arrow.convert");
        py::object from_arrays = vaex_arrow_convert.attr("list_from_arrays");
        return from_arrays(offsets_array, values_array);
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) {
        auto data_ptr = this->data_ptr[thread];
        auto data_ptr2 = this->data_ptr2[thread];
        auto data_mask_ptr = this->data_mask_ptr[thread];
        // auto data_mask_ptr2 = this->data_mask_ptr2[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];
        // auto grid_data_order = &this->grid_data_order[grid * this->grid->length1d];
        auto null_count = &this->null_count[grid * this->grid->length1d];
        auto nan_count = &this->nan_count[grid * this->grid->length1d];

        if (data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        for (size_t j = 0; j < length; j++) {
            IndexType i = indices1d[j];
            if ((data_mask_ptr == nullptr) || (data_mask_ptr[j]) == 1) {
                DataType value = data_ptr[offset + j];
                // DataType2 value_order = data_ptr2 == nullptr ? offset + j : data_ptr2[offset + j];
                if (FlipEndian) {
                    value = _to_native(value);
                    // value_order = _to_native(value_order);
                }
                // if (value == value && value_order == value_order) { // nan check
                if (value == value) { // nan check
                    grid_data[i].push_back(value);
                } else if (!dropnan) {
                    nan_count[i] += 1;
                }
            } else if ((data_mask_ptr != nullptr) && (data_mask_ptr[j] == 0) && (!dropnull)) {
                null_count[i] += 1;
            }
        }
    }
    int64_t *nan_count;
    int64_t *null_count;

    std::vector<data_type2 *> data_ptr2;
    std::vector<uint64_t> data_size2;
    std::vector<uint8_t *> data_mask_ptr2;
    std::vector<uint64_t> data_mask_size2;
    bool dropnan;
    bool dropnull;
};

template <class DataType = std::string, class DataType2 = double, class IndexType = default_index_type, bool FlipEndian = false>
class AggListString : public AggBaseString<StringList64, IndexType> {
  public:
    using Base = AggBaseString<StringList64, IndexType>;
    using Base::Base;
    using data_type = DataType;
    using data_type2 = DataType2;

    AggListString(Grid<IndexType> *grid, int grids, int threads, bool dropnan, bool dropnull)
        : Base(grid, grids, threads), data_ptr2(threads), data_size2(threads), dropnan(dropnan), dropnull(dropnull) {
        if (grids != 1) {
            throw std::runtime_error("list aggregation only accepts 1 grid");
        }
        null_count = new int64_t[this->count()];
    }
    void initial_fill(int grid) { std::fill(null_count + this->grid->length1d * grid, null_count + this->grid->length1d * (grid + 1), 0); }
    virtual ~AggListString() { delete[] null_count; }

    virtual void merge(std::vector<Aggregator *> others) {}
    virtual pybind11::object get_result() {
        std::shared_ptr<StringList64> sl = std::make_shared<StringList64>();
        py::array_t<int64_t> offsets_array;
        {
            if (!this->grid_used[0]) {
                this->initial_fill(0);
            }
            // we only have a single grid
            auto grid_data = &this->grid_data[0];
            offsets_array = py::array_t<int64_t>(this->grid->length1d + 1);
            int64_t *offsets = offsets_array.mutable_data(0);
            int64_t offset = 0;
            offsets[0] = 0;
            // first we fill the offsets
            for (size_t j = 0; j < this->grid->length1d; j++) {
                offset += grid_data[j].length;
                offset += this->null_count[j];
                offsets[j + 1] = offset;
            }

            py::gil_scoped_release release;
            int64_t flat_length = offset;
            for (size_t j = 0; j < this->grid->length1d; j++) {
                // for (auto &s : grid_data[j]) {
                for (int64_t i = 0; i < grid_data[j].length; i++) {
                    if (grid_data[j].is_null(i)) {
                        sl->push_null();
                    } else {
                        string_view s = grid_data[j].view(i);
                        sl->push(s);
                    }
                }
                // for (int64_t i = 0; i < this->null_count[j]; i++) {
                //     sl->push_null();
                // }
                // std::copy(grid_data[j].begin(), grid_data[j].end(), values + offsets[j]);
                // std::fill(values + offsets[j] + grid_data[j].size(), values + offsets[j] + grid_data[j].size() + this->nan_count[j], std::numeric_limits<data_type_fixed>::quiet_NaN());
            }
        }
        py::object vaex_arrow_convert = py::module::import("vaex.arrow.convert");
        py::object from_arrays = vaex_arrow_convert.attr("list_from_arrays");
        return from_arrays(offsets_array, sl);
    }
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) {
        auto string_sequence = this->string_sequence[thread];
        if (string_sequence == nullptr) {
            throw std::runtime_error("string_sequence not set");
        }
        // auto data_ptr = this->data_ptr[thread];
        auto data_ptr2 = this->data_ptr2[thread];
        auto data_mask_ptr = this->data_mask_ptr[thread];
        // auto data_mask_ptr2 = this->data_mask_ptr2[thread];
        auto grid_data = &this->grid_data[grid * this->grid->length1d];
        // auto grid_data_order = &this->grid_data_order[grid * this->grid->length1d];

        auto null_count = &this->null_count[grid * this->grid->length1d];
        auto nan_count = &this->nan_count[grid * this->grid->length1d];

        for (size_t j = 0; j < length; j++) {
            IndexType i = indices1d[j];
            if (!string_sequence->is_null(j + offset)) {
                // grid_data[i].push_back(string_sequence->get(j + offset));
                grid_data[i].push(string_sequence->view(j + offset));
            } else if (!dropnull) {
                grid_data[i].push_null();
            }
        }
    }
    int64_t *nan_count;
    int64_t *null_count;

    std::vector<data_type2 *> data_ptr2;
    std::vector<uint64_t> data_size2;
    std::vector<uint8_t *> data_mask_ptr2;
    std::vector<uint64_t> data_mask_size2;
    bool dropnan;
    bool dropnull;
};

void add_agg_list_string(py::module &m, py::class_<Aggregator> &base) {
    using T2 = int64_t;
    std::string class_name = std::string("AggList_string");
    class_name += "_";
    class_name += type_name<T2>::value;
    using Class = AggListString<>;
    py::class_<Class>(m, class_name.c_str(), base)
        .def(py::init<Grid<> *, int, int, bool, bool>(), py::keep_alive<1, 2>())
        .def("set_data", &Class::set_data)
        .def("clear_data_mask", &Class::clear_data_mask)
        .def("set_data_mask", &Class::set_data_mask)

        ;
}

template <class T, class T2, bool FlipEndian>
void add_agg_list_primitive_mixed(py::module &m, const py::class_<Aggregator> &base) {
    std::string class_name = std::string("AggList_");
    class_name += type_name<T>::value;
    class_name += "_";
    class_name += type_name<T2>::value;
    class_name += FlipEndian ? "_non_native" : "";
    using Class = AggListPrimitive<T, T2, default_index_type, FlipEndian>;
    py::class_<Class>(m, class_name.c_str(), base).def(py::init<Grid<> *, int, int, bool, bool>(), py::keep_alive<1, 2>());
}

template <class T, bool FlipEndian>
void add_agg_list_primitive(py::module &m, const py::class_<Aggregator> &base) {
    // add_agg_list_primitive_mixed<std::string, type, FlipEndian>(m, base);
    add_agg_list_primitive_mixed<T, int64_t, FlipEndian>(m, base);
}
#define create(type)                                                                                                                                                                                   \
    template void add_agg_list_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base);                                                                                              \
    template void add_agg_list_primitive<type, false>(py::module & m, const py::class_<Aggregator> &base);
#include "create_alltypes.hpp"

// for when we want to add a sort column
// template <class T, bool FlipEndian>
// void add_agg_list_primitive(py::module &m, const py::class_<Aggregator> &base) {
//     // add_agg_list_primitive_mixed<std::string, type, FlipEndian>(m, base);
// #define create(type) add_agg_list_primitive_mixed<T, type, FlipEndian>(m, base);
// #include "create_alltypes.hpp"
// }

// #undef create
// #define create(type) \
//     template void add_agg_list_primitive<type, true>(py::module & m, const py::class_<Aggregator> &base); \ template void add_agg_list_primitive<type, false>(py::module & m, const
//     py::class_<Aggregator> &base);
// #include "create_alltypes.hpp"

} // namespace vaex
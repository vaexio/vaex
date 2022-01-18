#include "agg.hpp"
#include "hash.hpp"
#include "superstring.hpp"

namespace vaex {

template <class T = uint64_t, class BinIndexType = default_index_type, bool FlipEndian = false>
class BinnerHash : public Binner {
    // format of bins if [invalid, bin0, bin1, ..., binN-1, out of range]
  public:
    using index_type = BinIndexType;
    using hash_type = hash_map<T>;
    BinnerHash(std::string expression, hash_type *hashmap)
        : Binner(expression), hashmap(hashmap), hash_bins(hashmap->size()), missing_bin(hashmap->null_index() + 1), nan_bin(hashmap->nan_index() + 1), data_ptr(nullptr), data_size(0),
          data_mask_ptr(nullptr) {}
    BinnerHash *copy() { return new BinnerHash(*this); }
    virtual ~BinnerHash() {}
    virtual void to_bins(uint64_t offset, index_type *output, uint64_t length, uint64_t stride) {
        std::vector<int64_t> bins;
        bins.resize(length);
        if (FlipEndian) {
            throw std::runtime_error("little endianness not implemented ");
        }
        hashmap->map_many(data_ptr, offset, length, &bins[0]);
        if (data_mask_ptr) {
            for (uint64_t i = offset; i < offset + length; i++) {
                // T value = data_ptr[i];
                // if (FlipEndian) {
                //     value = _to_native<>(value);
                // }
                // index_type bin = hashmap->map_key(value);
                index_type bin = bins[i - offset];
                index_type index = 0;
                // this followes numpy, 1 is masked
                bool masked = data_mask_ptr[i] == 1;
                // if (value != value) { // nan
                //     index = nan_bin;
                // } else
                if (masked) { // missing
                    index = missing_bin;
                } else if (bin < 0) { // invalid
                    index = 0;
                } else if (bin >= hash_bins) {
                    index = hash_bins + 2;
                } else {
                    index = bin + 1; // bins starts at 3
                }
                output[i - offset] += index * stride;
            }
        } else {
            for (uint64_t i = offset; i < offset + length; i++) {
                // T value = data_ptr[i];
                // if (FlipEndian) {
                //     value = _to_native<>(value);
                // }
                // index_type bin = hashmap->map_key(value);
                index_type bin = bins[i - offset];
                index_type index = 0;
                // if (value != value) { // nan
                // index = nan_bin;
                // } else
                if (bin < 0) { // invalid
                    index = 0;
                } else if (bin >= hash_bins) {
                    index = hash_bins + 2;
                } else {
                    index = bin + 1; // bins starts at 3
                }
                output[i - offset] += index * stride;
            }
        }
    } // namespace vaex
    virtual uint64_t size() const { return data_size; }
    virtual uint64_t shape() const { return hash_bins + 2; }
    void set_data(py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if (info.itemsize != sizeof(T)) {
            throw std::runtime_error("Itemsize of data and binner are not equal");
        }
        this->data_ptr = (T *)info.ptr;
        this->data_size = info.shape[0];
    }
    // pybind11 likes casting too much, this can slow things down
    // void set_data(py::array_t<T, py::array::c_style> ar) {
    //     auto m = ar.template mutable_unchecked<1>();
    //     this->ptr = &m(0);
    //     this->_size = ar.size();
    // }
    void clear_data_mask() {
        this->data_mask_ptr = nullptr;
        this->data_mask_size = 0;
    }
    void set_data_mask(py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr = (uint8_t *)info.ptr;
        this->data_mask_size = info.shape[0];
    }
    hash_type *hashmap;
    uint64_t hash_bins;
    uint64_t missing_bin;
    uint64_t nan_bin;
    T *data_ptr;
    uint64_t data_size;
    uint8_t *data_mask_ptr;
    uint64_t data_mask_size;
};

template <>
class BinnerHash<std::string, default_index_type, false> : public Binner {
    // format of bins if [invalid, bin0, bin1, ..., binN-1, out of range]
  public:
    using index_type = default_index_type;
    using T = std::string;
    using hash_type = hash_map<T>;
    BinnerHash(std::string expression, hash_type *hashmap) : Binner(expression), hashmap(hashmap), hash_bins(hashmap->size()), missing_bin(hashmap->null_index() + 1), strings(nullptr) {}
    BinnerHash *copy() { return new BinnerHash(*this); }
    virtual ~BinnerHash() {}
    virtual void to_bins(uint64_t offset, index_type *output, uint64_t length, uint64_t stride) {
        std::vector<int64_t> bins;
        bins.resize(length);
        hashmap->map_many(strings, offset, length, &bins[0]);
        for (uint64_t i = offset; i < offset + length; i++) {
            index_type index = 0;
            index_type bin = bins[i - offset];
            if (bin < 0) { // invalid
                index = 0;
            } else if (bin >= hash_bins) {
                index = hash_bins + 2;
            } else {
                index = bin + 1; // bins starts at 3
            }
            output[i - offset] += index * stride;
        }
    }
    void set_data_mask() {}
    void clear_data_mask() {}
    virtual uint64_t size() const { return this->strings ? this->strings->length : 0; }
    virtual uint64_t shape() const { return hash_bins + 2; }
    virtual void set_data(StringSequence *strings) { this->strings = strings; }
    hash_type *hashmap;
    uint64_t hash_bins;
    uint64_t missing_bin;
    StringSequence *strings;
};

template <class T, class Base, class Module, bool FlipEndian = false>
void add_binner_hash_(Module m, Base &base, std::string postfix) {
    typedef BinnerHash<T, default_index_type, FlipEndian> Type;
    std::string class_name = "BinnerHash_" + postfix;
    py::class_<Type>(m, class_name.c_str(), base)
        .def(py::init<std::string, typename Type::hash_type *>(), py::keep_alive<1, 3>()) // this keeps hashmap aline
        .def("set_data", &Type::set_data)
        .def("clear_data_mask", &Type::clear_data_mask)
        .def("set_data_mask", &Type::set_data_mask)
        .def("copy", &Type::copy)
        .def("__len__", [](const Type &binner) { return binner.shape(); })
        .def_property_readonly("expression", [](const Type &binner) { return binner.expression; })
        .def_property_readonly("hash_bins", [](const Type &binner) { return binner.hash_bins; })
        .def(py::pickle(
            [](const Type &binner) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(binner.expression, binner.hashmap);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                Type binner(t[0].cast<std::string>(), t[1].cast<typename Type::hash_type *>());
                return binner;
            }));
    ;
}

template <class T, class Base, class Module>
void add_binner_hash(Module m, Base &base, std::string postfix) {
    add_binner_hash_<T, Base, Module, false>(m, base, postfix);
    add_binner_hash_<T, Base, Module, true>(m, base, postfix + "_non_native");
}

void add_binner_hash(py::module &m, py::class_<Binner> &binner) {
    add_binner_hash<double>(m, binner, "float64");
    add_binner_hash<float>(m, binner, "float32");
    add_binner_hash<int64_t>(m, binner, "int64");
    add_binner_hash<int32_t>(m, binner, "int32");
    add_binner_hash<int16_t>(m, binner, "int16");
    add_binner_hash<int8_t>(m, binner, "int8");
    add_binner_hash<uint64_t>(m, binner, "uint64");
    add_binner_hash<uint32_t>(m, binner, "uint32");
    add_binner_hash<uint16_t>(m, binner, "uint16");
    add_binner_hash<uint8_t>(m, binner, "uint8");
    add_binner_hash<bool>(m, binner, "bool");
    add_binner_hash_<std::string>(m, binner, "string");
}

} // namespace vaex
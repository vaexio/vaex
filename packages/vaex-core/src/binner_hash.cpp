#include "agg.hpp"
#include "hash.hpp"
#include "superstring.hpp"
#include "utils.hpp"

namespace vaex {

template <class T = uint64_t, class BinIndexType = default_index_type, bool FlipEndian = false>
class BinnerHash : public Binner {
    // format of bins if [invalid, bin0, bin1, ..., binN-1, out of range]
  public:
    using index_type = BinIndexType;
    using hash_type = hash_map<T>;
    BinnerHash(int threads, std::string expression, hash_type *hashmap)
        : Binner(threads, expression), hashmap(hashmap), hash_bins(hashmap->size()), missing_bin(hashmap->null_index() + 1), nan_bin(hashmap->nan_index() + 1), data_ptr(threads), data_size(threads),
          data_mask_ptr(threads), data_mask_size(threads), bin_buffers(threads) {
        for (auto &buffer : bin_buffers) {
            buffer.resize(INDEX_BLOCK_SIZE);
        }
    }
    BinnerHash *copy() { return new BinnerHash(*this); }
    virtual ~BinnerHash() {}
    virtual void to_bins(int thread, uint64_t offset, index_type *output, uint64_t length, uint64_t stride) {
        auto data_ptr = this->data_ptr[thread];
        auto data_mask_ptr = this->data_mask_ptr[thread];
        int64_t *bins = &(this->bin_buffers[thread][0]);
        std::vector<typename FixBool<T>::value> flipped;
        if (FlipEndian) {
            flipped.resize(length);
            for (uint64_t i = offset; i < offset + length; i++) {
                T value = data_ptr[i];
                flipped[i - offset] = _to_native<>(value);
            }
            data_ptr = (T *)&flipped[0];
        }

        hashmap->map_many(data_ptr, offset, length, &bins[0]);

        if (data_mask_ptr) {
            for (uint64_t i = offset; i < offset + length; i++) {
                index_type bin = bins[i - offset];
                index_type index = 0;
                // this follows numpy, 1 is masked
                bool masked = data_mask_ptr[i] == 1;
                if (masked) { // missing
                    index = missing_bin;
                } else if (bin < 0) { // invalid
                    index = 0;
                } else if (bin >= hash_bins) {
                    index = hash_bins + 2;
                } else {
                    index = bin + 1; // bins starts at 1
                }
                output[i - offset] += index * stride;
            }
        } else {
            for (uint64_t i = offset; i < offset + length; i++) {
                index_type bin = bins[i - offset];
                index_type index = 0;
                if (bin < 0) { // invalid
                    index = 0;
                } else if (bin >= hash_bins) {
                    index = hash_bins + 2;
                } else {
                    index = bin + 1; // bins starts at 1
                }
                output[i - offset] += index * stride;
            }
        }
    } // namespace vaex
    virtual uint64_t data_length(int thread) const { return data_size[thread]; };
    virtual uint64_t shape() const { return hash_bins + 2; }
    void set_data(int thread, py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if (info.itemsize != sizeof(T)) {
            throw std::runtime_error("Itemsize of data and binner are not equal");
        }
        this->data_ptr[thread] = (T *)info.ptr;
        this->data_size[thread] = info.shape[0];
    }
    void set_data_mask(int thread, py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr[thread] = (uint8_t *)info.ptr;
        this->data_mask_size[thread] = info.shape[0];
    }
    void clear_data_mask(int thread) {
        this->data_mask_ptr[thread] = nullptr;
        this->data_mask_size[thread] = 0;
    }
    hash_type *hashmap;
    uint64_t hash_bins;
    uint64_t missing_bin;
    uint64_t nan_bin;
    std::vector<T *> data_ptr;
    std::vector<uint64_t> data_size;
    std::vector<uint8_t *> data_mask_ptr;
    std::vector<uint64_t> data_mask_size;
    std::vector<std::vector<int64_t>> bin_buffers;
};

template <>
class BinnerHash<std::string, default_index_type, false> : public Binner {
    // format of bins if [invalid, bin0, bin1, ..., binN-1, out of range]
  public:
    using index_type = default_index_type;
    using T = string_ref;
    using hash_type = hash_map<T>;
    BinnerHash(int threads, std::string expression, hash_type *hashmap)
        : Binner(threads, expression), hashmap(hashmap), hash_bins(hashmap->size()), missing_bin(hashmap->null_index() + 1), strings(threads) {}
    BinnerHash *copy() { return new BinnerHash(*this); }
    virtual ~BinnerHash() {}
    virtual void to_bins(int thread, uint64_t offset, index_type *output, uint64_t length, uint64_t stride) {
        std::vector<int64_t> bins;
        bins.resize(length);
        auto strings = this->strings[thread];
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
    void set_data_mask(int thread) {}
    void clear_data_mask(int thread) {}
    virtual uint64_t data_length(int thread) const { return strings[thread]->length; };
    virtual uint64_t shape() const { return hash_bins + 2; }
    virtual void set_data(int thread, StringSequence *strings) { this->strings[thread] = strings; }
    hash_type *hashmap;
    uint64_t hash_bins;
    uint64_t missing_bin;
    std::vector<StringSequence *> strings;
};

template <class T, bool FlipEndian = false>
void add_binner_hash_(py::module &m, py::class_<Binner> &base, std::string postfix) {
    typedef BinnerHash<T, default_index_type, FlipEndian> Type;
    std::string class_name = "BinnerHash_" + postfix;
    py::class_<Type>(m, class_name.c_str(), base)
        .def(py::init<int, std::string, typename Type::hash_type *>(), py::keep_alive<1, 4>()) // this keeps hashmap alive
        .def("set_data", &Type::set_data)
        .def("clear_data_mask", &Type::clear_data_mask)
        .def("set_data_mask", &Type::set_data_mask)
        .def("copy", &Type::copy)
        .def("__len__", [](const Type &binner) { return binner.shape(); })
        .def_property_readonly("expression", [](const Type &binner) { return binner.expression; })
        .def_property_readonly("hash_bins", [](const Type &binner) { return binner.hash_bins; })
        .def(py::pickle(
            [](const Type &binner) { // __getstate__
                return py::make_tuple(binner.threads, binner.expression, binner.hashmap);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");
                Type binner(t[0].cast<int>(), t[1].cast<std::string>(), t[2].cast<typename Type::hash_type *>());
                return binner;
            }));
    ;
}

template <class T>
void add_binner_hash(py::module &m, py::class_<Binner> &base) {
    std::string postfix(type_name<T>::value);
    add_binner_hash_<T, false>(m, base, postfix);
    add_binner_hash_<T, true>(m, base, postfix + "_non_native");
}

void add_binner_hash(py::module &m, py::class_<Binner> &binner) {
#define create(type) add_binner_hash<type>(m, binner);
#include "create_alltypes.hpp"
    add_binner_hash_<std::string>(m, binner, "string");
}

} // namespace vaex
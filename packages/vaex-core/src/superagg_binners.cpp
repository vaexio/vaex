#include "agg.hpp"

namespace vaex {

template <class T = double, class BinIndexType = default_index_type, bool FlipEndian = false>
class BinnerScalar : public Binner {
  public:
    using index_type = BinIndexType;
    BinnerScalar(std::string expression, double vmin, double vmax, uint64_t bins) : Binner(expression), vmin(vmin), vmax(vmax), bins(bins), data_mask_ptr(nullptr) {}
    BinnerScalar *copy() { return new BinnerScalar(*this); }
    virtual ~BinnerScalar() {}
    virtual void to_bins(uint64_t offset, index_type *output, uint64_t length, uint64_t stride) {
        const double scale_v = 1. / (vmax - vmin);
        if (data_mask_ptr) {
            for (uint64_t i = offset; i < offset + length; i++) {
                T value = ptr[i];
                if (FlipEndian) {
                    value = _to_native<>(value);
                }
                double value_double = value;
                double scaled = (value_double - vmin) * scale_v;
                index_type index = 0;
                bool masked = data_mask_ptr[i] == 1;
                if (scaled != scaled || masked) { // nan goes to index 0
                } else if (scaled < 0) {          // smaller values are put at offset 1
                    index = 1;
                } else if (scaled >= 1) { // bigger values are put at offset -1 (last)
                    index = bins - 1 + 3;
                } else {
                    index = (int)(scaled * (bins)) + 2; // real data starts at 2
                }
                output[i - offset] += index * stride;
            }
        } else {
            for (uint64_t i = offset; i < offset + length; i++) {
                T value = ptr[i];
                if (FlipEndian) {
                    value = _to_native<>(value);
                }
                double value_double = value;
                double scaled = (value_double - vmin) * scale_v;
                index_type index = 0;
                if (scaled != scaled) {  // nan goes to index 0
                } else if (scaled < 0) { // smaller values are put at offset 1
                    index = 1;
                } else if (scaled >= 1) { // bigger values are put at offset -1 (last)
                    index = bins - 1 + 3;
                } else {
                    index = (int)(scaled * (bins)) + 2; // real data starts at 2
                }
                output[i - offset] += index * stride;
            }
        }
    }
    virtual uint64_t size() const { return _size; }
    virtual uint64_t shape() const { return bins + 3; }
    void set_data(py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if (info.itemsize != sizeof(T)) {
            throw std::runtime_error("Itemsize of data and binner are not equal");
        }
        this->ptr = (T *)info.ptr;
        this->_size = info.shape[0];
    }
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
    double vmin;
    double vmax;
    uint64_t bins;
    T *ptr;
    uint64_t _size;
    uint8_t *data_mask_ptr;
    uint64_t data_mask_size;
};

template <class T, class Base, class Module, bool FlipEndian>
void add_binner_scalar_(Module m, Base &base, std::string postfix) {
    typedef BinnerScalar<T, default_index_type, FlipEndian> Type;
    std::string class_name = "BinnerScalar_" + postfix;
    py::class_<Type>(m, class_name.c_str(), base)
        .def(py::init<std::string, double, double, uint64_t>())
        .def("set_data", &Type::set_data)
        .def("clear_data_mask", &Type::clear_data_mask)
        .def("set_data_mask", &Type::set_data_mask)
        .def("copy", &Type::copy)
        .def("__len__", [](const Type &binner) { return binner.bins + 3; })
        .def_property_readonly("expression", [](const Type &binner) { return binner.expression; })
        .def_property_readonly("bins", [](const Type &binner) { return binner.bins; })
        .def_property_readonly("vmin", [](const Type &binner) { return binner.vmin; })
        .def_property_readonly("vmax", [](const Type &binner) { return binner.vmax; })
        .def(py::pickle(
            [](const Type &binner) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(binner.expression, binner.vmin, binner.vmax, binner.bins);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");
                Type binner(t[0].cast<std::string>(), t[1].cast<T>(), t[2].cast<T>(), t[3].cast<uint64_t>());
                return binner;
            }));
    ;
}

template <class T, class Base, class Module>
void add_binner_scalar(Module m, Base &base, std::string postfix) {
    add_binner_scalar_<T, Base, Module, false>(m, base, postfix);
    add_binner_scalar_<T, Base, Module, true>(m, base, postfix + "_non_native");
}

py::object binners_to_1d(int64_t length, std::vector<Binner *> binners) {
    using IndexType = uint64_t;
    py::array_t<IndexType, py::array::c_style> indices1d_ar(length);
    IndexType *indices1d = indices1d_ar.mutable_data(0);
    {
        py::gil_scoped_release gil;
        int64_t dimensions = binners.size();
        uint64_t *shapes = new uint64_t[dimensions];
        uint64_t *strides = new uint64_t[dimensions];
        // uint64_t length1d = 1;
        for (size_t i = 0; i < dimensions; i++) {
            shapes[i] = binners[i]->shape();
            // length1d *= shapes[i];
            printf("shape: %i\n", shapes[i]);
        }
        if (dimensions > 0) {
            strides[0] = 1;
            for (size_t i = 1; i < dimensions; i++) {
                strides[i] = strides[i - 1] * shapes[i - 1];
                printf("stride: %i\n", strides[i]);
            }
        }
        std::fill(indices1d, indices1d + length, 0);
        for (size_t i = 0; i < dimensions; i++) {
            binners[i]->to_bins(0, indices1d, length, strides[i]);
        }
        delete[] strides;
        delete[] shapes;
    }
    return indices1d_ar;
}

void add_binner_ordinal(py::module &m, py::class_<Binner> &binner);
void add_binner_hash(py::module &m, py::class_<Binner> &binner);
void add_binner_combined(py::module &m, py::class_<Binner> &binner);

void add_binners(py::module &m, py::class_<Binner> &binner) {

    add_binner_ordinal(m, binner);
    add_binner_hash(m, binner);
    add_binner_combined(m, binner);
    add_binner_scalar<double>(m, binner, "float64");
    add_binner_scalar<float>(m, binner, "float32");
    add_binner_scalar<int64_t>(m, binner, "int64");
    add_binner_scalar<int32_t>(m, binner, "int32");
    add_binner_scalar<int16_t>(m, binner, "int16");
    add_binner_scalar<int8_t>(m, binner, "int8");
    add_binner_scalar<uint64_t>(m, binner, "uint64");
    add_binner_scalar<uint32_t>(m, binner, "uint32");
    add_binner_scalar<uint16_t>(m, binner, "uint16");
    add_binner_scalar<uint8_t>(m, binner, "uint8");
    add_binner_scalar<bool>(m, binner, "bool");

    m.def("binners_to_1d", binners_to_1d);
}
} // namespace vaex
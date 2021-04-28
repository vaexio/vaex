#include "agg.hpp"

namespace vaex {

template<class T=double, class BinIndexType=default_index_type, bool FlipEndian=false>
class BinnerScalar : public Binner {
public:
    using index_type = BinIndexType;
    BinnerScalar(std::string expression, double vmin, double vmax, uint64_t bins) : Binner(expression), vmin(vmin), vmax(vmax), bins(bins), data_mask_ptr(nullptr) { }
    BinnerScalar* copy() { 
        return new BinnerScalar(*this);
    }
    virtual ~BinnerScalar() { }
    virtual void to_bins(uint64_t offset, index_type* output, uint64_t length, uint64_t stride) {
        const double scale_v = 1./ (vmax-vmin);
        if(data_mask_ptr) {
            for(uint64_t i = offset; i < offset + length; i++) {
                T value = ptr[i];
                if(FlipEndian) {
                    value = _to_native<>(value);
                }
                double value_double = value;
                double scaled = (value_double - vmin) * scale_v;
                index_type index = 0;
                bool masked = data_mask_ptr[i] == 1;
                if(scaled != scaled || masked) { // nan goes to index 0                
                } else if (scaled < 0) { // smaller values are put at offset 1
                    index = 1;
                } else if (scaled >= 1) { // bigger values are put at offset -1 (last)
                    index = bins-1+3;
                } else {
                    index = (int)(scaled * (bins)) + 2; // real data starts at 2
                }
                output[i-offset] += index * stride;
            }
        } else {
            for(uint64_t i = offset; i < offset + length; i++) {
                T value = ptr[i];
                if(FlipEndian) {
                    value = _to_native<>(value);
                }
                double value_double = value;
                double scaled = (value_double - vmin) * scale_v;
                index_type index = 0;
                if(scaled != scaled) { // nan goes to index 0                
                } else if (scaled < 0) { // smaller values are put at offset 1
                    index = 1;
                } else if (scaled >= 1) { // bigger values are put at offset -1 (last)
                    index = bins-1+3;
                } else {
                    index = (int)(scaled * (bins)) + 2; // real data starts at 2
                }
                output[i-offset] += index * stride;
            }
        }
    }
    virtual uint64_t size() {
        return _size;
    }
    virtual uint64_t shape() {
        return bins + 3;
    }
    void set_data(py::buffer ar) {
        py::buffer_info info = ar.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if(info.itemsize != sizeof(T)) {
            throw std::runtime_error("Itemsize of data and binner are not equal");
        }
        this->ptr = (T*)info.ptr;
        this->_size = info.shape[0];
    }
    void clear_data_mask() {
        this->data_mask_ptr = nullptr;
        this->data_mask_size = 0;
    }
    void set_data_mask(py::buffer ar) {
        py::buffer_info info = ar.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr = (uint8_t*)info.ptr;
        this->data_mask_size = info.shape[0];
    }
    double vmin;
    double vmax;
    uint64_t bins;
    T* ptr;
    uint64_t _size;
    uint8_t* data_mask_ptr;
    uint64_t data_mask_size;
};

template<class T=uint64_t, class BinIndexType=default_index_type, bool FlipEndian=false>
class BinnerOrdinal : public Binner {
public:
    using index_type = BinIndexType;
    BinnerOrdinal(std::string expression, uint64_t ordinal_count, uint64_t min_value=0) : Binner(expression), ordinal_count(ordinal_count), min_value(min_value), data_mask_ptr(nullptr) { }
    BinnerOrdinal* copy() { 
        return new BinnerOrdinal(*this);
    }
    virtual ~BinnerOrdinal() { }
    virtual void to_bins(uint64_t offset, index_type* output, uint64_t length, uint64_t stride) {
        if(data_mask_ptr) {
            for(uint64_t i = offset; i < offset + length; i++) {
                T value = ptr[i] - min_value;
                if(FlipEndian) {
                    value = _to_native<>(value);
                }
                index_type index = 0;
                // this followes numpy, 1 is masked
                bool masked = data_mask_ptr[i] == 1;
                if(value != value || masked) { // nan goes to index 0                
                } else if (value < 0) { // smaller values are put at offset 1
                    index = 1;
                } else if (value >= ordinal_count) { // bigger values are put at offset -1 (last)
                    index = ordinal_count-1+3;
                } else {
                    index = value + 2; // real data starts at 2
                }
                output[i-offset] += index * stride;
            }
        } else {
            for(uint64_t i = offset; i < offset + length; i++) {
                T value = ptr[i] - min_value;
                if(FlipEndian) {
                    value = _to_native<>(value);
                }
                index_type index = 0;
                if(value != value) { // nan goes to index 0                
                } else if (value < 0) { // smaller values are put at offset 1
                    index = 1;
                } else if (value >= ordinal_count) { // bigger values are put at offset -1 (last)
                    index = ordinal_count-1+3;
                } else {
                    index = value + 2; // real data starts at 2
                }
                output[i-offset] += index * stride;
            }
        }
    }
    virtual uint64_t size() {
        return _size;
    }
    virtual uint64_t shape() {
        return ordinal_count + 3;
    }
    void set_data(py::buffer ar) {
        py::buffer_info info = ar.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if(info.itemsize != sizeof(T)) {
            throw std::runtime_error("Itemsize of data and binner are not equal");
        }
        this->ptr = (T*)info.ptr;
        this->_size = info.shape[0];
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
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr = (uint8_t*)info.ptr;
        this->data_mask_size = info.shape[0];
    }
    uint64_t ordinal_count;
    uint64_t min_value;
    T* ptr;
    uint64_t _size;
    uint8_t* data_mask_ptr;
    uint64_t data_mask_size;
};

template<class T, class Base, class Module, bool FlipEndian>
void add_binner_ordinal_(Module m, Base& base, std::string postfix) { 
    typedef BinnerOrdinal<T, default_index_type, FlipEndian> Type;
    std::string class_name = "BinnerOrdinal_" + postfix;
    py::class_<Type>(m, class_name.c_str(), base)
        .def(py::init<std::string, T, T>())
        .def("set_data", &Type::set_data)
        .def("clear_data_mask", &Type::clear_data_mask)
        .def("set_data_mask", &Type::set_data_mask)
        .def("copy", &Type::copy)
        .def_property_readonly("expression", [](const Type &binner) {
                return binner.expression;
            }
        )
        .def_property_readonly("ordinal_count", [](const Type &binner) {
                return binner.ordinal_count;
            }
        )
        .def_property_readonly("min_value", [](const Type &binner) {
                return binner.min_value;
            }
        )
    ;
}

template<class T, class Base, class Module>
void add_binner_ordinal(Module m, Base& base, std::string postfix) { 
    add_binner_ordinal_<T, Base, Module, false>(m, base, postfix);
    add_binner_ordinal_<T, Base, Module, true>(m, base, postfix+"_non_native");
}


template<class T, class Base, class Module, bool FlipEndian>
void add_binner_scalar_(Module m, Base& base, std::string postfix) {
    typedef BinnerScalar<T, default_index_type, FlipEndian> Type;
    std::string class_name = "BinnerScalar_" + postfix;
    py::class_<Type>(m, class_name.c_str(), base)
        .def(py::init<std::string, double, double, uint64_t>())
        .def("set_data", &Type::set_data)
        .def("clear_data_mask", &Type::clear_data_mask)
        .def("set_data_mask", &Type::set_data_mask)
        .def("copy", &Type::copy)
        .def_property_readonly("expression", [](const Type &binner) {
                return binner.expression;
            }
        )
        .def_property_readonly("bins", [](const Type &binner) {
                return binner.bins;
            }
        )
        .def_property_readonly("vmin", [](const Type &binner) {
                return binner.vmin;
            }
        )
        .def_property_readonly("vmax", [](const Type &binner) {
                return binner.vmax;
            }
        )
    ;
}


template<class T, class Base, class Module>
void add_binner_scalar(Module m, Base& base, std::string postfix) {
    add_binner_scalar_<T, Base, Module, false>(m, base, postfix);
    add_binner_scalar_<T, Base, Module, true>(m, base, postfix+"_non_native");
}


void add_binners(py::module &m, py::class_<Binner>& binner) {
    add_binner_ordinal<double>(m, binner, "float64");
    add_binner_ordinal<float>(m, binner, "float32");
    add_binner_ordinal<int64_t>(m, binner, "int64");
    add_binner_ordinal<int32_t>(m, binner, "int32");
    add_binner_ordinal<int16_t>(m, binner, "int16");
    add_binner_ordinal<int8_t>(m, binner, "int8");
    add_binner_ordinal<uint64_t>(m, binner, "uint64");
    add_binner_ordinal<uint32_t>(m, binner, "uint32");
    add_binner_ordinal<uint16_t>(m, binner, "uint16");
    add_binner_ordinal<uint8_t>(m, binner, "uint8");
    add_binner_ordinal<bool>(m, binner, "bool");

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
}
} // namespace vaex
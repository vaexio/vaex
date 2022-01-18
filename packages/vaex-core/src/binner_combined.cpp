#include "agg.hpp"

namespace vaex {

template <class T = uint64_t, class BinIndexType = default_index_type, bool FlipEndian = false>
class BinnerCombined : public Binner {
  public:
    using index_type = BinIndexType;
    BinnerCombined(std::vector<Binner *> binners) : Binner(""), binners(binners), dimensions(binners.size()) {
        dimensions = binners.size();
        strides.resize(dimensions);
        shapes.resize(dimensions);
        // length1d = 1;
        for (size_t i = 0; i < dimensions; i++) {
            shapes[i] = binners[i]->shape();
            // length1d *= shapes[i];
        }
        if (dimensions > 0) {
            strides[0] = 1;
            for (size_t i = 1; i < dimensions; i++) {
                strides[i] = strides[i - 1] * shapes[i - 1];
            }
        }
    }
    BinnerCombined *copy() { return new BinnerCombined(*this); }
    virtual ~BinnerCombined() {}
    virtual void to_bins(uint64_t offset, index_type *output, uint64_t length, uint64_t stride) {
        for (int64_t i = 0; i < dimensions; i++) {
            binners[i]->to_bins(offset, output, length, stride * strides[i]);
        }
    }
    virtual uint64_t size() const { return 0; }
    virtual uint64_t shape() const { return shapes[dimensions - 1]; }
    std::vector<Binner *> binners;
    int64_t dimensions;
    std::vector<int64_t> strides;
    std::vector<int64_t> shapes;
};

void add_binner_combined(py::module &m, py::class_<Binner> &binner) {
    typedef BinnerCombined<> Type;
    std::string class_name = "BinnerCombined";
    py::class_<Type>(m, class_name.c_str())
        .def(py::init<std::vector<Binner *>>(), py::keep_alive<1, 2>()) // this keeps the binner alive
        // .def("set_data", &Type::set_data)
        // .def("clear_data_mask", &Type::clear_data_mask)
        // .def("set_data_mask", &Type::set_data_mask)
        .def("copy", &Type::copy)
        .def("__len__", [](const Type &binner) { return binner.shape(); })
        // .def_property_readonly("expression", [](const Type &binner) { return binner.expression; })
        // .def_property_readonly("hash_bins", [](const Type &binner) { return binner.hash_bins; })
        .def_property_readonly("binners", [](const Type &binner) { return binner.binners; })
        .def(py::pickle(
            [](const Type &binner) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(binner.expression, binner.binners);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                Type binner(t[0].cast<std::vector<Binner *>>());
                return binner;
            }));
    ;
}

} // namespace vaex
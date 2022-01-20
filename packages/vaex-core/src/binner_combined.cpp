#include "agg.hpp"

namespace vaex {

template <class T = uint64_t, class BinIndexType = default_index_type, bool FlipEndian = false>
class BinnerCombined : public Binner {
  public:
    using index_type = BinIndexType;
    BinnerCombined(int threads, std::vector<Binner *> binners) : Binner(threads, ""), binners(binners), dimensions(binners.size()) {
        dimensions = binners.size();
        strides.resize(dimensions);
        shapes.resize(dimensions);
        for (int64_t i = 0; i < dimensions; i++) {
            shapes[i] = binners[i]->shape();
        }
        if (dimensions > 0) {
            strides[0] = 1;
            for (int64_t i = 1; i < dimensions; i++) {
                strides[i] = strides[i - 1] * shapes[i - 1];
            }
        }
    }
    BinnerCombined *copy() { return new BinnerCombined(*this); }
    virtual ~BinnerCombined() {}
    virtual void to_bins(int thread, uint64_t offset, index_type *output, uint64_t length, uint64_t stride) {
        for (int64_t i = 0; i < dimensions; i++) {
            binners[i]->to_bins(thread, offset, output, length, stride * strides[i]);
        }
    }
    virtual uint64_t data_length(int thread) const { return this->binners[0]->data_length(thread); }
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
        .def(py::init<int, std::vector<Binner *>>(), py::keep_alive<1, 3>()) // this keeps the binner alive
        .def("copy", &Type::copy)
        .def("__len__", [](const Type &binner) { return binner.shape(); })
        .def_property_readonly("binners", [](const Type &binner) { return binner.binners; })
        .def(py::pickle(
            [](const Type &binner) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(binner.threads, binner.binners);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                Type binner(t[0].cast<int>(), t[1].cast<std::vector<Binner *>>());
                return binner;
            }));
    ;
}

} // namespace vaex
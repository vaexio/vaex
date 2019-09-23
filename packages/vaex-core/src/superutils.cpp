#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>

namespace py = pybind11;

#define custom_isnan(value) (!(value==value))

namespace vaex {
    void init_hash_primitives(py::module &);
    void init_hash_string(py::module &);
    void init_hash_object(py::module &);
}

class Mask {
public:
    Mask(size_t length) : length(length), _owns_data(true) {
        mask_data = new uint8_t[length];
        reset();
    }
    Mask(uint8_t* mask_data, size_t length) : mask_data(mask_data), length(length), _owns_data(false) {
    }
    virtual ~Mask() {
        if(_owns_data)
            delete[] mask_data;
    }
    std::pair<int64_t, int64_t> indices(int64_t i1, int64_t i2) {
        if(i2 < i1) {
            throw std::runtime_error("end index should be larger or equal to start index");
        }
        int64_t count = 0;
        int64_t start = -1, end = -1;
        for(int64_t i = 0; i < length; i++) {
            if(mask_data[i] == 1) {
                if(count == i1) {
                    start = i;
                }
                if(count == i2) {
                    end = i;
                }
                count++;
            }
        }
        return {start, end};
    }
    void reset() {
        py::gil_scoped_release release;
        std::fill(mask_data, mask_data+length, 2);
    }
    int64_t count() {
        py::gil_scoped_release release;
        int64_t count = 0;
        for(int64_t i = 0; i < length; i++) {
            if(mask_data[i] == 1) {
                count++;
            }
        }
        return count;
    }
    int64_t is_dirty() {
        py::gil_scoped_release release;
        for(int64_t i = 0; i < length; i++) {
            if(mask_data[i] == 2) {
                return true;
            }
        }
        return false;
    }
    Mask* view(int64_t start, int64_t end) {
        if(end < start) {
            throw std::runtime_error("end index should be larger or equal to start index");
        }
        if(start < 0) {
            throw std::runtime_error("start should be >= 0");
        }
        if(end > length) {
            throw std::runtime_error("end should be <= length");
        }
        return new Mask(mask_data+start, end-start);
    }
    py::array_t<int64_t> first(int64_t amount) {
        auto ar = py::array_t<int64_t>(amount);
        auto ar_unsafe = ar.mutable_unchecked<1>();
        int64_t found = 0;
        {
            py::gil_scoped_release release;
            for(size_t i = 0; i < length; i++) {
                if(mask_data[i] == 1) {
                    ar_unsafe(found++) = i;
                }
                if(found == amount) {
                    break;
                }
            }
        }
        auto ar_trimmed = py::array_t<int64_t>(found);
        auto ar_trimmed_unsafe = ar_trimmed.mutable_unchecked<1>();
        for(size_t i = 0; i < found; i++) {
            ar_trimmed_unsafe(i) = ar_unsafe(i);
        }
        return ar_trimmed;
    }
    py::array_t<int64_t> last(int64_t amount) {
        auto ar = py::array_t<int64_t>(amount);
        auto ar_unsafe = ar.mutable_unchecked<1>();
        int64_t found = 0;
        {
            py::gil_scoped_release release;
            for(int64_t i = length-1; i >= 0; i--) {
                if(mask_data[i] == 1) {
                    ar_unsafe(found++) = i;
                }
                if(found == amount) {
                    break;
                }
            }
        }
        auto ar_ordered = py::array_t<int64_t>(found);
        auto ar_ordered_unsafe = ar_ordered.mutable_unchecked<1>();
        for(size_t i = 0; i < found; i++) {
            ar_ordered_unsafe(i) = ar_unsafe(found-1-i);
        }
        return ar_ordered;
    }
    uint8_t* mask_data;
    int64_t length;
    bool _owns_data;
};

PYBIND11_MODULE(superutils, m) {
    _import_array();

    m.doc() = "fast utils";

    py::class_<Mask>(m, "Mask", py::buffer_protocol())
        .def(py::init<size_t>())
        .def_buffer([](Mask &mask) -> py::buffer_info {
            std::vector<ssize_t> strides = {1};
            std::vector<ssize_t> shapes = {mask.length};
            return py::buffer_info(
                (void*)mask.mask_data,                               /* Pointer to buffer */
                sizeof(bool),                 /* Size of one scalar */
                py::format_descriptor<bool>::format(), /* Python struct-style format descriptor */
                1,                       /* Number of dimensions */
                shapes,                 /* Buffer dimensions */
                strides
            );
        })
        .def_property_readonly("length", [](const Mask &mask) {
                return mask.length;
            }
        )
        .def("indices", &Mask::indices)
        .def("count", &Mask::count)
        .def("first", &Mask::first)
        .def("last", &Mask::last)
        .def("reset", &Mask::reset)
        .def("is_dirty", &Mask::is_dirty)
        .def("view", &Mask::view, py::keep_alive<0, 1>())
        // .def("reduce", &Mask::reduce)
    ;


    vaex::init_hash_primitives(m);
    vaex::init_hash_string(m);
    vaex::init_hash_object(m);
}

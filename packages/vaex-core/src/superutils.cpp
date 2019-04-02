#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>

namespace py = pybind11;

#define custom_isnan(value) (!(value==value))


template<class T>
py::object unique(py::array_t<T> values, bool return_inverse=false) {
    auto m = values.template unchecked<1>();
    if(return_inverse) {
        int64_t size = m.size();
        std::unordered_map<T, int64_t> mapping;
        std::vector<T> ordered_uniques;
        py::array_t<int64_t> indices_array(size);
        auto indices = indices_array.mutable_unchecked<1>();
        bool has_nan = false;
        int64_t unique_size = 1; // we start counting at 1, and use 0 for nan
        {
            py::gil_scoped_release release;
            for(int64_t i = 0; i < size; i++) {
                T value = m(i);
                if(custom_isnan(value)) {
                    has_nan = true;
                    indices[i] = 0;
                } else {
                    auto search = mapping.find(value);
                    if(search != mapping.end()) {
                        indices[i] = search->second;
                    } else {
                        indices[i] = unique_size;
                        ordered_uniques.push_back(value);
                        mapping.emplace(value, unique_size++);
                    }
                }
            }
        }
        if(has_nan) unique_size++;
        py::array_t<T> unique_values(unique_size);
        auto u = unique_values.template mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            if(has_nan) {
                u[0] = NAN;
                T* ptr = &u[1];
                std::copy(ordered_uniques.begin(), ordered_uniques.end(), ptr);

            } else {
                T* ptr = &u[0];
                std::copy(ordered_uniques.begin(), ordered_uniques.end(), ptr);
                // no nan, so correct the indices
                for(int64_t i = 0; i < size; i++) {
                    indices[i]--;
                }
            }
        }
        // if(return_inverse)
        return py::make_tuple(unique_values, indices_array);
    } else {
        std::set<T> unique_set;
        bool has_nan = false;
        int64_t size = m.size();
        {
            py::gil_scoped_release release;
            for(int64_t i = 0; i < size; i++) {
                if(custom_isnan(m(i))) {
                    has_nan = true;
                } else {
                    unique_set.insert(m(i));
                    // std::cout << "found " << m(i) << std::endl;
                }
            }
        }
        int64_t unique_size = unique_set.size();
        if(has_nan) unique_size++;
        py::array_t<T> unique_values(unique_size);
        auto u = unique_values.template mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            if(has_nan) {
                u[0] = NAN;
                T* ptr = &u[1];
                std::copy(unique_set.begin(), unique_set.end(), ptr);
            } else {
                T* ptr = &u[0];
                std::copy(unique_set.begin(), unique_set.end(), ptr);
            }
        }
        return std::move(unique_values);
    }
}

void init_hash_primitives(py::module &);
void init_hash_string(py::module &);
void init_hash_object(py::module &);


PYBIND11_MODULE(superutils, m) {
    _import_array();

    m.doc() = "fast utils";
    m.def("unique", &unique<double>, "Find unique elements", py::arg("values"), py::arg("return_inverse")=false);
    m.def("unique", &unique<float>, "Find unique elements", py::arg("values"), py::arg("return_inverse")=false);
    init_hash_primitives(m);
    init_hash_string(m);
    init_hash_object(m);
}

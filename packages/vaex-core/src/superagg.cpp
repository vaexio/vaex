#include "agg.hpp"
#include <stdint.h>
#include <limits>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include "superstring.hpp"

using namespace vaex;



template<class GridType=uint64_t, class IndexType=default_index_type>
class AggBaseObject : public AggregatorBase<IndexType> {
public:
    using Base = AggregatorBase<IndexType>;
    using typename Base::index_type;
    using data_type = PyObject*;
    AggBaseObject(Grid<IndexType>* grid) : Base(grid), objects(nullptr), data_mask_ptr(nullptr) {
    }
    ~AggBaseObject() {
    }
    void set_data(py::buffer ar, size_t index) {
        py::buffer_info info = ar.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if("O" != info.format) {
            std::string msg = "Expected object type";
            throw std::runtime_error(msg);
        }
        this->objects = (data_type*)info.ptr;
        this->objects_size = info.shape[0];
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
    data_type* objects;
    uint8_t* data_mask_ptr;
    uint64_t data_mask_size;
    uint64_t objects_size;
};

template<class GridType=uint64_t, class IndexType=default_index_type>
class AggObjectCount : public AggBaseObject<GridType, IndexType> {
public:
    using Base = AggBaseObject<GridType, IndexType>;
    using Type = AggObjectCount<GridType, IndexType>;
    using Base::Base;
    virtual void reduce(std::vector<Type*> others) {
        for(auto other: others) {
            for(size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] += other->grid_data[i];
            }
        }
    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->objects == nullptr) {
            throw std::runtime_error("object data not set");
        }
        if(this->data_mask_ptr == nullptr) {
            for(size_t j = 0; j < length; j++) {
                PyObject* obj = this->objects[j+offset];
                bool none = (obj == Py_None);
                bool _isnan = PyFloat_Check(obj) && std::isnan(PyFloat_AsDouble(obj));
                this->grid_data[indices1d[j]] += (none || _isnan ? 0 : 1);
            }
        } else {
            for(size_t j = 0; j < length; j++) {
                PyObject* obj = this->objects[j+offset];
                bool none = (obj == Py_None);
                bool _isnan = PyFloat_Check(obj) && std::isnan(PyFloat_AsDouble(obj));
                bool masked = this->data_mask_ptr[j+offset] == 0;
                this->grid_data[indices1d[j]] += (none || masked || _isnan ? 0 : 1);
            }
        }
    }
    virtual bool can_release_gil() {
        return false;
    };
};



template<class GridType=uint64_t, class IndexType=default_index_type>
class AggStringCount : public AggBaseString<GridType, IndexType> {
public:
    using Base = AggBaseString<GridType, IndexType>;
    using Type = AggStringCount<GridType, IndexType>;
    using Base::Base;
    virtual void reduce(std::vector<Type*> others) {
        for(auto other: others) {
            for(size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] += other->grid_data[i];
            }
        }
    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->string_sequence == nullptr) {
            throw std::runtime_error("string_sequence not set");
        }
        if(!this->string_sequence->has_null() && this->data_mask_ptr == nullptr) {
            // fast path
            for(size_t j = 0; j < length; j++) {
                this->grid_data[indices1d[j]] += 1;
            }
        } else if(this->string_sequence->has_null() && this->data_mask_ptr == nullptr) {
            for(size_t j = 0; j < length; j++) {
                this->grid_data[indices1d[j]] += this->string_sequence->is_null(j+offset) ? 0 : 1;
            }
        } else if(!this->string_sequence->has_null() && this->data_mask_ptr != nullptr) {
            for(size_t j = 0; j < length; j++) {
                bool masked = this->data_mask_ptr[j+offset] == 0;
                this->grid_data[indices1d[j]] += masked ? 0 : 1;
            }
        } else if(this->string_sequence->has_null() && this->data_mask_ptr != nullptr) {
            for(size_t j = 0; j < length; j++) {
                bool masked = this->data_mask_ptr[j+offset] == 0;
                this->grid_data[indices1d[j]] += this->string_sequence->is_null(j+offset) || masked ? 0 : 1;
            }
        }
    }
};

template<class DataType=double, class GridType=DataType, class IndexType=default_index_type>
class AggBase : public AggregatorBase<GridType, IndexType> {
public:
    using Base = AggregatorBase<GridType, IndexType>;
    using typename Base::index_type;
    using data_type = DataType;
    AggBase(Grid<IndexType>* grid) : Base(grid), data_ptr(nullptr), data_mask_ptr(nullptr) {
    }
    void set_data(py::buffer ar, size_t index) {
        py::buffer_info info = ar.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_ptr = (data_type*)info.ptr;
        this->data_size = info.shape[0];
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
    data_type* data_ptr;
    uint64_t data_size;
    uint8_t* data_mask_ptr;
    uint64_t data_mask_size;
};

template<class StorageType=double, class IndexType=default_index_type, bool FlipEndian=false>
class AggCount : public AggBase<StorageType, int64_t, IndexType> {
public:
    using Base = AggBase<StorageType, int64_t, IndexType>;
    using Type = AggCount<StorageType, IndexType, FlipEndian>;
    using Base::Base;
    virtual void reduce(std::vector<Type*> others) {
        for(auto other: others) {
            for(size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] += other->grid_data[i];
            }
        }
    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {

        // }
        if(this->data_mask_ptr || this->data_ptr) {
            for(size_t j = 0; j < length; j++) {
                // if not masked
                if(this->data_mask_ptr == nullptr || this->data_mask_ptr[j+offset] == 1) {
                    // and not nan (TODO: we can skip this for non-floats)
                    if(this->data_ptr) {
                        StorageType value = this->data_ptr[j+offset];
                        if(FlipEndian)
                            value = _to_native(value);
                        if(value != value) // nan
                            continue;
                    }
                    this->grid_data[indices1d[j]] += 1;
                }
            }
        } else {
            for(size_t j = 0; j < length; j++) {
                this->grid_data[indices1d[j]] += 1;
            }
        }
    }
};

template<class StorageType=double, class IndexType=default_index_type, bool FlipEndian=false>
class AggMax : public AggBase<StorageType, StorageType, IndexType> {
public:
    using Base = AggBase<StorageType, StorageType, IndexType>;
    using Type = AggMax<StorageType, IndexType, FlipEndian>;
    using Base::Base;
    AggMax(Grid<IndexType>* grid) : Base(grid){
        typedef std::numeric_limits<StorageType> limit_type;
        // TODO: avoid double fill, since we also call it in the base ctor
        StorageType fill_value = limit_type::has_infinity ? -limit_type::infinity() : limit_type::min();
        std::fill(this->grid_data, this->grid_data+this->grid->length1d, fill_value);
    }
    virtual void reduce(std::vector<Type*> others) {
        for(auto other: others) {
            for(size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] = std::max(this->grid_data[i], other->grid_data[i]);
            }
        }
    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        if(this->data_mask_ptr) {
            for(size_t j = 0; j < length; j++) {
                // if not masked
                if(this->data_mask_ptr[j+offset] == 1) {
                    StorageType value = this->data_ptr[j+offset];
                    if(FlipEndian)
                        value = _to_native(value);
                    if(value != value) // nan
                        continue;
                    this->grid_data[indices1d[j]] = std::max(value, this->grid_data[indices1d[j]]);
                }
            }
        } else {
            for(size_t j = 0; j < length; j++) {
                StorageType value = this->data_ptr[offset + j];
                if(FlipEndian)
                    value = _to_native(value);
                if(value == value) // nan check
                    this->grid_data[indices1d[j]] = std::max(value, this->grid_data[indices1d[j]]);
            }
        }
    }
};

template<class StorageType=double, class IndexType=default_index_type, bool FlipEndian=false>
class AggMin : public AggBase<StorageType, StorageType, IndexType> {
public:
    using Base = AggBase<StorageType, StorageType, IndexType>;
    using Type = AggMin<StorageType, IndexType, FlipEndian>;
    using Base::Base;
    AggMin(Grid<IndexType>* grid) : Base(grid){
        typedef std::numeric_limits<StorageType> limit_type;
        StorageType fill_value = limit_type::has_infinity ? limit_type::infinity() : limit_type::max();
        // TODO: avoid double fill, since we also call it in the base ctor
        std::fill(this->grid_data, this->grid_data+this->grid->length1d, fill_value);
    }
    virtual void reduce(std::vector<Type*> others) {
        for(auto other: others) {
            for(size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] = std::min(this->grid_data[i], other->grid_data[i]);
            }
        }
    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }

        if(this->data_mask_ptr) {
            for(size_t j = 0; j < length; j++) {
                // if not masked
                if(this->data_mask_ptr[j+offset] == 1) {
                    StorageType value = this->data_ptr[j+offset];
                    if(FlipEndian)
                        value = _to_native(value);
                    if(value != value) // nan
                        continue;
                    this->grid_data[indices1d[j]] = std::min(value, this->grid_data[indices1d[j]]);
                }
            }
        } else {
            for(size_t j = 0; j < length; j++) {
                StorageType value = this->data_ptr[offset + j];
                if(FlipEndian)
                    value = _to_native(value);
                if(value == value) // nan check
                    this->grid_data[indices1d[j]] = std::min(value, this->grid_data[indices1d[j]]);
            }
        }
    }
};

template<class T>
struct upcast {
};

template<>
struct upcast<float> {
    typedef double type;
};

template<>
struct upcast<double> {
    typedef double type;
};

template<>
struct upcast<bool> {
    typedef int64_t type;
};

template<>
struct upcast<int8_t> {
    typedef int64_t type;
};

template<>
struct upcast<int16_t> {
    typedef int64_t type;
};

template<>
struct upcast<int32_t> {
    typedef int64_t type;
};

template<>
struct upcast<int64_t> {
    typedef int64_t type;
};

template<>
struct upcast<uint8_t> {
    typedef uint64_t type;
};

template<>
struct upcast<uint16_t> {
    typedef uint64_t type;
};

template<>
struct upcast<uint32_t> {
    typedef uint64_t type;
};

template<>
struct upcast<uint64_t> {
    typedef uint64_t type;
};


template<class StorageType=double, class IndexType=default_index_type, bool FlipEndian=false>
class AggSum : public AggBase<StorageType, typename upcast<StorageType>::type, IndexType> {
public:
    using Base = AggBase<StorageType, typename upcast<StorageType>::type, IndexType>;
    using Type = AggSum<StorageType, IndexType, FlipEndian>;
    using Base::Base;
    virtual void reduce(std::vector<Type*> others) {
        for(auto other: others) {
            for(size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] = this->grid_data[i] + other->grid_data[i];
            }
        }
    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }

        if(this->data_mask_ptr) {
            for(size_t j = 0; j < length; j++) {
                // if not masked
                if(this->data_mask_ptr[j+offset] == 1) {
                    StorageType value = this->data_ptr[j+offset];
                    if(FlipEndian)
                        value = _to_native(value);
                    if(value != value) // nan
                        continue;
                    this->grid_data[indices1d[j]] += value;
                }
            }
        } else {
            for(size_t j = 0; j < length; j++) {
                StorageType value = this->data_ptr[offset + j];
                if(FlipEndian)
                    value = _to_native(value);
                if(value == value) // nan check
                    this->grid_data[indices1d[j]] += value;
            }
        }
    }
};

template<class StorageType=double, class IndexType=default_index_type, bool FlipEndian=false>
class AggSumMoment : public AggBase<StorageType, typename upcast<StorageType>::type, IndexType> {
public:
    using Base = AggBase<StorageType, typename upcast<StorageType>::type, IndexType>;
    using Type = AggSumMoment<StorageType, IndexType, FlipEndian>;
    using Base::Base;
    AggSumMoment(Grid<IndexType>* grid, uint32_t moment) : Base(grid), moment(moment) {
    }
    virtual void reduce(std::vector<Type*> others) {
        for(auto other: others) {
            for(size_t i = 0; i < this->grid->length1d; i++) {
                this->grid_data[i] = this->grid_data[i] + other->grid_data[i];
            }
        }
    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }

        if(this->data_mask_ptr) {
            for(size_t j = 0; j < length; j++) {
                // if not masked
                if(this->data_mask_ptr[j+offset] == 1) {
                    typename Base::grid_type value = this->data_ptr[j+offset];
                    if(FlipEndian)
                        value = _to_native(value);
                    if(value != value) // nan
                        continue;
                    this->grid_data[indices1d[j]] += pow(value, moment);
                }
            }
        } else {
            for(size_t j = 0; j < length; j++) {
                typename Base::grid_type value = this->data_ptr[offset + j];
                if(FlipEndian)
                    value = _to_native(value);
                if(value == value) // nan check
                    this->grid_data[indices1d[j]] += pow(value, moment);
            }
        }
    }
    size_t moment;
};

template<class StorageType=double, class IndexType=default_index_type, bool FlipEndian=false>
class AggFirst : public AggBase<StorageType, StorageType, IndexType> {
public:
    using Base = AggBase<StorageType, StorageType, IndexType>;
    using Type = AggFirst<StorageType, IndexType, FlipEndian>;
    using Base::Base;
    AggFirst(Grid<IndexType>* grid) : Base(grid) {
        grid_data_order = (StorageType*)malloc(sizeof(StorageType) * grid->length1d);
        typedef std::numeric_limits<StorageType> limit_type;
        std::fill(grid_data_order, grid_data_order+grid->length1d, limit_type::max());
    }
    virtual ~AggFirst() {
        free(grid_data_order);
    }
    void set_data(py::buffer ar, size_t index) {
        py::buffer_info info = ar.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if(index == 1) {
            this->data_ptr2 = (StorageType*)info.ptr;
            this->data_size2 = info.shape[0];
        } else {
            this->data_ptr = (StorageType*)info.ptr;
            this->data_size = info.shape[0];
        }
    }
    void set_data_mask2(py::buffer ar) {
        py::buffer_info info = ar.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr2 = (uint8_t*)info.ptr;
        this->data_mask_size2 = info.shape[0];
    }
    virtual void reduce(std::vector<Type*> others) {
        for(auto other: others) {
            for(size_t i = 0; i < this->grid->length1d; i++) {
                if(other->grid_data_order[i] < this->grid_data_order[i]) {
                    this->grid_data[i] = other->grid_data[i];
                    this->grid_data_order[i] = other->grid_data_order[i];
                }
            }
        }
    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        if(this->data_ptr2 == nullptr) {
            throw std::runtime_error("data2 not set");
        }
        // TODO: masked support
        for(size_t j = 0; j < length; j++) {
            StorageType value = this->data_ptr[offset + j];
            StorageType value_order = this->data_ptr2[offset + j];
            if(FlipEndian) {
                value = _to_native(value);
                value_order = _to_native(value_order);
            }
            if(value == value && value_order == value_order) { // nan check
                IndexType i = indices1d[j];
                if(value_order < grid_data_order[i]) {
                    this->grid_data[i] = value;
                    this->grid_data_order[i] = value_order;
                }
            }
        }

    }
    StorageType* grid_data_order;        
    StorageType* data_ptr2;
    uint64_t data_size2;
    uint8_t* data_mask_ptr2;
    uint64_t data_mask_size2;
};


template<class Agg, class Base, class Module>
void add_agg(Module m, Base& base, const char* class_name) {
    py::class_<Agg>(m, class_name, py::buffer_protocol(), base)
        .def(py::init<Grid<>*>(), py::keep_alive<1, 2>())
        .def_buffer([](Agg &agg) -> py::buffer_info {
            std::vector<ssize_t> strides(agg.grid->dimensions);
            std::vector<ssize_t> shapes(agg.grid->dimensions);
            std::copy(&agg.grid->shapes[0], &agg.grid->shapes[agg.grid->dimensions], &shapes[0]);
            std::transform(&agg.grid->strides[0], &agg.grid->strides[agg.grid->dimensions], &strides[0], [](uint64_t x) { return x*sizeof(typename Agg::grid_type); } );
            return py::buffer_info(
                agg.grid_data,                               /* Pointer to buffer */
                sizeof(typename Agg::grid_type),                 /* Size of one scalar */
                py::format_descriptor<typename Agg::grid_type>::format(), /* Python struct-style format descriptor */
                agg.grid->dimensions,                       /* Number of dimensions */
                shapes,                 /* Buffer dimensions */
                strides
            );
        })
        .def_property_readonly("grid", [](const Agg &agg) {
                return agg.grid;
            }
        )
        .def("set_data", &Agg::set_data)
        .def("set_data_mask", &Agg::set_data_mask)
        .def("clear_data_mask", &Agg::clear_data_mask)
        .def("reduce", &Agg::reduce)
    ;
}


template<class Agg, class Base, class Module, class A>
void add_agg_arg(Module m, Base& base, const char* class_name) {
    py::class_<Agg>(m, class_name, py::buffer_protocol(), base)
        .def(py::init<Grid<>*, A>(), py::keep_alive<1, 2>())
        .def_buffer([](Agg &agg) -> py::buffer_info {
            std::vector<ssize_t> strides(agg.grid->dimensions);
            std::vector<ssize_t> shapes(agg.grid->dimensions);
            std::copy(&agg.grid->shapes[0], &agg.grid->shapes[agg.grid->dimensions], &shapes[0]);
            std::transform(&agg.grid->strides[0], &agg.grid->strides[agg.grid->dimensions], &strides[0], [](uint64_t x) { return x*sizeof(typename Agg::grid_type); } );
            return py::buffer_info(
                agg.grid_data,                               /* Pointer to buffer */
                sizeof(typename Agg::grid_type),                 /* Size of one scalar */
                py::format_descriptor<typename Agg::grid_type>::format(), /* Python struct-style format descriptor */
                agg.grid->dimensions,                       /* Number of dimensions */
                shapes,                 /* Buffer dimensions */
                strides
            );
        })
        .def_property_readonly("grid", [](const Agg &agg) {
                return agg.grid;
            }
        )
        .def("set_data", &Agg::set_data)
        .def("set_data_mask", &Agg::set_data_mask)
        .def("clear_data_mask", &Agg::clear_data_mask)
        .def("reduce", &Agg::reduce)
    ;
}

template<class T, class Base, class Module, bool FlipEndian=false>
void add_agg_primitives_(Module m, Base& base, std::string postfix) {
    add_agg<AggCount<T, default_index_type, FlipEndian>, Base, Module>(m, base, ("AggCount_" + postfix).c_str());
    add_agg<AggMin<T, default_index_type, FlipEndian>, Base, Module>(m, base, ("AggMin_" + postfix).c_str());
    add_agg<AggMax<T, default_index_type, FlipEndian>, Base, Module>(m, base, ("AggMax_" + postfix).c_str());
    add_agg<AggSum<T, default_index_type, FlipEndian>, Base, Module>(m, base, ("AggSum_" + postfix).c_str());
    add_agg<AggFirst<T, default_index_type, FlipEndian>, Base, Module>(m, base, ("AggFirst_" + postfix).c_str());
    add_agg_arg<AggSumMoment<T, default_index_type, FlipEndian>, Base, Module, uint32_t>(m, base, ("AggSumMoment_" + postfix).c_str());
}

template<class T, class Base, class Module>
void add_agg_primitives(Module m, Base& base, std::string postfix) {
    add_agg_primitives_<T, Base, Module, false>(m, base, postfix);
    add_agg_primitives_<T, Base, Module, true>(m, base, postfix+ "_non_native");
}




namespace vaex {
    void add_agg_nunique_string(py::module& m, py::class_<Aggregator>& base);
    void add_agg_nunique_primitives(py::module& m, py::class_<Aggregator>& base);
    void add_binners(py::module &, py::class_<Binner>& base);
};

PYBIND11_MODULE(superagg, m) {
    _import_array();

    m.doc() = "fast statistics/aggregation on grids";
    py::class_<Aggregator> aggregator(m, "Aggregator");
    py::class_<AggregatorBase<>> agg(m, "Agg", aggregator);
    py::class_<Binner> binner(m, "Binner");

    {
        typedef Grid<> Type;
        py::class_<Type>(m, "Grid")
            .def(py::init<std::vector<Binner*> >(), py::keep_alive<1, 2>())
            .def("bin", (void (Type::*)(std::vector<Aggregator*>, size_t))&Type::bin)
            .def("bin", (void (Type::*)(std::vector<Aggregator*> ))&Type::bin)
            .def_property_readonly("binners", [](const Type &grid) {
                    return grid.binners;
                }
            )
        ;
    }

    vaex::add_agg_nunique_string(m, aggregator);
    vaex::add_agg_nunique_primitives(m, aggregator);
    vaex::add_binners(m, binner);
    add_agg<AggStringCount<>>(m, agg, "AggCount_string");
    add_agg<AggObjectCount<>>(m, agg, "AggCount_object");
    add_agg_primitives<double>(m, agg, "float64");
    add_agg_primitives<float>(m, agg, "float32");
    add_agg_primitives<int64_t>(m, agg, "int64");
    add_agg_primitives<int32_t>(m, agg, "int32");
    add_agg_primitives<int16_t>(m, agg, "int16");
    add_agg_primitives<int8_t>(m, agg, "int8");
    add_agg_primitives<uint64_t>(m, agg, "uint64");
    add_agg_primitives<uint32_t>(m, agg, "uint32");
    add_agg_primitives<uint16_t>(m, agg, "uint16");
    add_agg_primitives<uint8_t>(m, agg, "uint8");
    add_agg_primitives<bool>(m, agg, "bool");

}

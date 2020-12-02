#include "agg.hpp"
#include "hash_primitives.hpp"

namespace vaex {

template<class DataType=double, class GridType=uint64_t, class IndexType=default_index_type, bool FlipEndian=false>
class AggNUnique : public Aggregator {
public:
    using Type = AggNUnique<DataType, GridType, IndexType, FlipEndian>;
    using Counter = counter<DataType, hashmap_primitive>; // TODO: do we want a prime growth variant?
    using index_type = IndexType;
    using grid_type = GridType;
    using data_type = DataType;
    AggNUnique(Grid<IndexType>* grid, bool dropmissing, bool dropnan) : grid(grid), grid_data(nullptr), data_ptr(nullptr), data_mask_ptr(nullptr), selection_mask_ptr(nullptr), dropmissing(dropmissing), dropnan(dropnan) {
        counters = new Counter[grid->length1d];
    }
    virtual ~AggNUnique() {
        if(grid_data)
            free(grid_data);
        delete[] counters;
    }
    virtual void reduce(std::vector<Type*> others) {
        if(grid_data == nullptr) {
            grid_data = (grid_type*)malloc(sizeof(grid_type) * grid->length1d);
        }
        for(size_t i = 0; i < this->grid->length1d; i++) {
            for(auto other: others) {
                this->counters[i].merge(other->counters[i]);
            }
            grid_data[i] = counters[i].map.size();
            if(!dropmissing)
                grid_data[i] += counters[i].null_count;
            if(!dropnan)
                grid_data[i] += counters[i].nan_count;
            
        }

    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->data_ptr == nullptr) {
            throw std::runtime_error("data not set");
        }
        for(size_t j = 0; j < length; j++) {
            bool masked = false;
            if(this->selection_mask_ptr && this->data_mask_ptr[j+offset] == 0)
                continue; // if value is not in selection/filter, don't even consider it
            if(this->data_mask_ptr && this->data_mask_ptr[j+offset] == 0)
                masked = true;
            if(masked) {
                this->counters[indices1d[j]].update1_null();
            } else {
                data_type value = this->data_ptr[j+offset];
                if(FlipEndian)
                    value = _to_native(value);
                if(value != value) // nan
                    this->counters[indices1d[j]].update1_nan();
                else
                    this->counters[indices1d[j]].update1(value);
            }
        }
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
    void set_selection_mask(py::buffer ar) {
        py::buffer_info info = ar.request();
        if(info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->selection_mask_ptr = (uint8_t*)info.ptr;
        this->selection_mask_size = info.shape[0];
    }
    Grid<IndexType>* grid;
    grid_type* grid_data;
    Counter* counters;
    data_type* data_ptr;
    uint64_t data_size;
    uint8_t* data_mask_ptr;
    uint64_t data_mask_size;
    uint8_t* selection_mask_ptr;
    uint64_t selection_mask_size;
    bool dropmissing;
    bool dropnan;
};


template<class Agg, class Base, class Module>
void add_agg(Module m, Base& base, const char* class_name) {
    py::class_<Agg>(m, class_name, py::buffer_protocol(), base)
        .def(py::init<Grid<>*, bool, bool>(), py::keep_alive<1, 2>())
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
        .def("clear_data_mask", &Agg::clear_data_mask)
        .def("set_data_mask", &Agg::set_data_mask)
        .def("set_selection_mask", &Agg::set_selection_mask)
        .def("reduce", &Agg::reduce)
    ;
}


template<class T, class Base, class Module, bool FlipEndian=false>
void add_agg_primitives_(Module m, Base& base, std::string postfix) {
    // add_agg<AggCount<T, default_index_type, FlipEndian>, Base, Module>(m, base, ("AggCount_" + postfix).c_str());
    add_agg<AggNUnique<T, uint64_t, default_index_type, FlipEndian>>(m, base, ("AggNUnique_" + postfix).c_str());
}
template<class T, class Base, class Module>
void add_agg_primitives(Module m, Base& base, std::string postfix) {
    add_agg_primitives_<T, Base, Module, false>(m, base, postfix);
    add_agg_primitives_<T, Base, Module, true>(m, base, postfix+ "_non_native");
}

void add_agg_nunique_primitives(py::module& m, py::class_<Aggregator>& base) {
    std::string postfix = "string";
    add_agg_primitives<double>(m, base, "float64");
    add_agg_primitives<float>(m, base, "float32");
    add_agg_primitives<int64_t>(m, base, "int64");
    add_agg_primitives<int32_t>(m, base, "int32");
    add_agg_primitives<int16_t>(m, base, "int16");
    add_agg_primitives<int8_t>(m, base, "int8");
    add_agg_primitives<uint64_t>(m, base, "uint64");
    add_agg_primitives<uint32_t>(m, base, "uint32");
    add_agg_primitives<uint16_t>(m, base, "uint16");
    add_agg_primitives<uint8_t>(m, base, "uint8");
    add_agg_primitives<bool>(m, base, "bool");

}

}
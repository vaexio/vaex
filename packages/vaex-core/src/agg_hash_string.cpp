#include "agg.hpp"
#include "hash_string.cpp"

namespace vaex {

template<class GridType=uint64_t, class IndexType=default_index_type>
class AggStringNUnique : public Aggregator {
public:
    using Type = AggStringNUnique<GridType, IndexType>;
    using index_type = IndexType;
    using grid_type = GridType;
    AggStringNUnique(Grid<IndexType>* grid, bool dropmissing, bool dropnan) : grid(grid), grid_data(nullptr), string_sequence(nullptr), data_mask_ptr(nullptr), selection_mask_ptr(nullptr), dropmissing(dropmissing), dropnan(dropnan) {
        counters = new counter<>[grid->length1d];
    }
    virtual ~AggStringNUnique() {
        if(grid_data)
            free(grid_data);
        delete[] counters;
    }
    virtual void reduce(std::vector<Type*> others) {
        if(grid_data == nullptr)
            grid_data = (grid_type*)malloc(sizeof(grid_type) * grid->length1d);
        for(size_t i = 0; i < this->grid->length1d; i++) {
            for(auto other: others) {
                this->counters[i].merge(other->counters[i]);
            }
            if(dropmissing)
                grid_data[i] = counters[i].map.size();
            else
                grid_data[i] = counters[i].map.size() + counters[i].null_count;
            
        }

    }
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) {
        if(this->string_sequence == nullptr) {
            throw std::runtime_error("string_sequence not set");
        }
        for(size_t j = 0; j < length; j++) {
            bool masked = false;
            if(this->selection_mask_ptr && this->data_mask_ptr[j+offset] == 0)
                continue; // if value is not in selection/filter, don't even consider it
            if(this->data_mask_ptr && this->data_mask_ptr[j+offset] == 0)
                masked = true;
            if(this->string_sequence->is_null(j+offset))
                masked = true;
            if(masked) {
                this->counters[indices1d[j]].update1_null();
            } else {
                string s = this->string_sequence->get(j+offset);
                this->counters[indices1d[j]].update1(s);
            }
        }
    }
    void set_data(StringSequence* string_sequence, size_t index) {
        this->string_sequence = string_sequence;
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
    counter<>* counters;
    StringSequence* string_sequence;
    uint8_t* data_mask_ptr;
    uint64_t data_mask_size;
    uint8_t* selection_mask_ptr;
    uint64_t selection_mask_size;
    bool dropmissing;
    bool dropnan; // not used for strings
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
            if(agg.grid_data == nullptr) {
                throw std::runtime_error("No grid_data");
            }
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

void add_agg_nunique_string(py::module& m, py::class_<Aggregator>& base) {
    std::string postfix = "string";
    add_agg<AggStringNUnique<>>(m, base, ("AggNUnique_" + postfix).c_str());
}


}
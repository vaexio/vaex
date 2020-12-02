#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "superstring.hpp"

namespace py = pybind11;

namespace vaex {


template<class T>
T _to_native(T value_non_native) {
	unsigned char* bytes = (unsigned char*)&value_non_native;
	T result;
	unsigned char* result_bytes = (unsigned char*)&result;
	for(size_t i = 0; i < sizeof(T); i++)
		result_bytes[sizeof(T)-1-i] = bytes[i];
	return result;
}


const int INDEX_BLOCK_SIZE = 1024;
const int MAX_DIM = 16;
typedef uint64_t default_index_type;


class Binner {
public:
    Binner(std::string expression) : expression(expression) { }
    virtual ~Binner() {}
    virtual void to_bins(uint64_t offset, default_index_type* output, uint64_t length, uint64_t stride) = 0;
    virtual uint64_t size() = 0;
    virtual uint64_t shape() = 0;
    std::string expression;
};

class Aggregator {
public:
    virtual ~Aggregator() {}
    virtual void aggregate(default_index_type* indices1d, size_t length, uint64_t offset) = 0;
    virtual bool can_release_gil() {
        return true;
    };
};

template<class IndexType=default_index_type>
class Grid {
public:
    using index_type = IndexType;
    Grid(std::vector<Binner*> binners) : binners(binners) {
        indices1d = (IndexType*)malloc(INDEX_BLOCK_SIZE * sizeof(IndexType));
        dimensions = binners.size();
        shapes = new uint64_t[dimensions];
        strides = new uint64_t[dimensions];
        length1d = 1;
        for(size_t i =  0; i < dimensions; i++) {
            shapes[i] = binners[i]->shape();
            length1d *= shapes[i];
        }
        if(dimensions > 0) {
            strides[0] = 1;
            for(size_t i = 1; i < dimensions; i++) {
                strides[i] = strides[i-1] * shapes[i-1];
            }
        }
    }
    virtual ~Grid() {
        free(indices1d);
        delete[] strides;
        delete[] shapes;
    }
    void bin(std::vector<Aggregator*> aggregators) {
        if(binners.size() == 0) {
            throw std::runtime_error("no binners set and no length given");
        } else {
            uint64_t length = binners[0]->size();
            this->bin(aggregators, length);
        }
    }
    void bin(std::vector<Aggregator*> aggregators, size_t length) {
            std::vector<Aggregator*> aggregators_no_gil;
            std::vector<Aggregator*> aggregators_gil;
            for(auto agg : aggregators) {
                if(agg->can_release_gil()) {
                    aggregators_no_gil.push_back(agg);
                } else {
                    aggregators_gil.push_back(agg);
                }
            }
            {
                if(aggregators_no_gil.size() > 0) {
                    py::gil_scoped_release release;
                    this->bin_(aggregators_no_gil, length);
                }
            }
            {
                if(aggregators_gil.size() > 0) {
                    this->bin_(aggregators_gil, length);
                }
            }
    }
    void bin_(std::vector<Aggregator*> aggregators, size_t length) {
        size_t binner_count = binners.size();
        size_t aggregator_count = aggregators.size();
        uint64_t offset = 0;
        bool done = false;
        while(!done) {
            uint64_t leftover = length - offset;
            if(leftover < INDEX_BLOCK_SIZE) {
                std::fill(indices1d, indices1d+leftover, 0);
                for(size_t i = 0; i < binner_count; i++) {
                    binners[i]->to_bins(offset, indices1d, leftover, this->strides[i]);
                }
            } else {
                std::fill(indices1d, indices1d+INDEX_BLOCK_SIZE, 0);
                for(size_t i = 0; i < binner_count; i++) {
                    binners[i]->to_bins(offset, indices1d, INDEX_BLOCK_SIZE, this->strides[i]);
                }
            }
            if(leftover < INDEX_BLOCK_SIZE) {
                for(size_t i = 0; i < aggregator_count; i++) {
                    aggregators[i]->aggregate(indices1d, leftover, offset);
                }
            } else {
                for(size_t i = 0; i < aggregator_count; i++) {
                    aggregators[i]->aggregate(indices1d, INDEX_BLOCK_SIZE, offset);
                }
            }
            offset += (leftover < INDEX_BLOCK_SIZE) ? leftover :  INDEX_BLOCK_SIZE;
            done = offset == length;
        }
    }
    std::vector<Binner*> binners;
    index_type *indices1d;
    uint64_t* strides;
    uint64_t* shapes;
    uint64_t dimensions;
    size_t length1d;
};

template<class GridType=double, class IndexType=default_index_type>
class AggregatorBase : public Aggregator {
public:
    using index_type = IndexType;
    using grid_type = GridType;
    AggregatorBase(Grid<IndexType>* grid, grid_type fill_value) : grid(grid) {
        grid_data = (grid_type*)malloc(sizeof(grid_type) * grid->length1d);
        std::fill(grid_data, grid_data+grid->length1d, fill_value);
    }
    AggregatorBase(Grid<IndexType>* grid) : grid(grid) {
        grid_data = (grid_type*)malloc(sizeof(grid_type) * grid->length1d);
        std::fill(grid_data, grid_data+grid->length1d, 0);
    }
    virtual ~AggregatorBase() {
        free(grid_data);
    }
    Grid<IndexType>* grid;
    grid_type* grid_data;
};

template<class GridType, class IndexType=default_index_type>
class AggregatorBaseCls : public Aggregator {
public:
    using index_type = IndexType;
    using grid_type = GridType;
    AggregatorBaseCls(Grid<IndexType>* grid) : grid(grid) {
        grid_data = new grid_type[grid->length1d];
    }
    virtual ~AggregatorBaseCls() {
        delete[] grid_data;
    }
    Grid<IndexType>* grid;
    grid_type* grid_data;
};

template<class GridType=uint64_t, class IndexType=default_index_type>
class AggBaseString : public AggregatorBase<IndexType> {
public:
    using Base = AggregatorBase<IndexType>;
    using typename Base::index_type;
    using data_type = StringSequence;
    AggBaseString(Grid<IndexType>* grid) : Base(grid), string_sequence(nullptr), data_mask_ptr(nullptr) {
    }
    ~AggBaseString() {
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
    StringSequence* string_sequence;
    uint8_t* data_mask_ptr;
    uint64_t data_mask_size;
};




}

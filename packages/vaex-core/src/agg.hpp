#pragma once
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "superstring.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace vaex {

template <class T>
T _to_native(T value_non_native) {
    unsigned char *bytes = (unsigned char *)&value_non_native;
    T result;
    unsigned char *result_bytes = (unsigned char *)&result;
    for (size_t i = 0; i < sizeof(T); i++)
        result_bytes[sizeof(T) - 1 - i] = bytes[i];
    return result;
}

const int INDEX_BLOCK_SIZE = 1024;
const int MAX_DIM = 16;
typedef uint64_t default_index_type;

class Binner {
  public:
    Binner(int threads, std::string expression) : threads(threads), expression(expression) {}
    virtual ~Binner() {}
    virtual void to_bins(int thread, uint64_t offset, default_index_type *output, uint64_t length, uint64_t stride) = 0;
    virtual uint64_t data_length(int thread) const = 0;
    virtual uint64_t shape() const = 0;
    int threads;
    std::string expression;
};

class Aggregator {
  public:
    virtual ~Aggregator() {}
    virtual void aggregate(int thread, default_index_type *indices1d, size_t length, uint64_t offset) = 0;
    virtual void merge(std::vector<Aggregator *>) = 0;
    virtual py::object get_result() = 0;
    virtual size_t bytes_used() = 0;
    virtual bool can_release_gil() { return true; };
};

template <class IndexType = default_index_type>
class Grid {
  public:
    using index_type = IndexType;
    Grid(std::vector<Binner *> binners) : binners(binners), dimensions(binners.size()), shapes(binners.size()), strides(binners.size()) {
        length1d = 1;
        for (size_t i = 0; i < dimensions; i++) {
            shapes[i] = binners[i]->shape();
            length1d *= shapes[i];
        }
        if (dimensions > 0) {
            strides[0] = 1;
            for (size_t i = 1; i < dimensions; i++) {
                strides[i] = strides[i - 1] * shapes[i - 1];
            }
        }
    }
    virtual ~Grid() {}
    void bin(int thread, std::vector<Aggregator *> aggregators) {
        if (binners.size() == 0) {
            throw std::runtime_error("no binners set and no length given");
        } else {
            uint64_t length = binners[0]->data_length(thread);
            this->bin(thread, aggregators, length);
        }
    }
    void bin(int thread, std::vector<Aggregator *> aggregators, size_t length) {
        std::vector<Aggregator *> aggregators_no_gil;
        std::vector<Aggregator *> aggregators_gil;
        for (auto agg : aggregators) {
            if (agg->can_release_gil()) {
                aggregators_no_gil.push_back(agg);
            } else {
                aggregators_gil.push_back(agg);
            }
        }
        {
            if (aggregators_no_gil.size() > 0) {
                py::gil_scoped_release release;
                this->bin_(thread, aggregators_no_gil, length);
            }
        }
        {
            if (aggregators_gil.size() > 0) {
                this->bin_(thread, aggregators_gil, length);
            }
        }
    }
    void bin_(int thread, std::vector<Aggregator *> aggregators, size_t length) {
        size_t binner_count = binners.size();
        size_t aggregator_count = aggregators.size();
        uint64_t offset = 0;
        bool done = false;
        std::vector<IndexType> indices1d(INDEX_BLOCK_SIZE);
        while (!done) {
            uint64_t leftover = length - offset;
            if (leftover < INDEX_BLOCK_SIZE) {
                std::fill(&indices1d[0], &indices1d[0] + leftover, 0);
                for (size_t i = 0; i < binner_count; i++) {
                    binners[i]->to_bins(thread, offset, &indices1d[0], leftover, this->strides[i]);
                }
            } else {
                std::fill(&indices1d[0], &indices1d[0] + INDEX_BLOCK_SIZE, 0);
                for (size_t i = 0; i < binner_count; i++) {
                    binners[i]->to_bins(thread, offset, &indices1d[0], INDEX_BLOCK_SIZE, this->strides[i]);
                }
            }
            if (leftover < INDEX_BLOCK_SIZE) {
                for (size_t i = 0; i < aggregator_count; i++) {
                    aggregators[i]->aggregate(thread, &indices1d[0], leftover, offset);
                }
            } else {
                for (size_t i = 0; i < aggregator_count; i++) {
                    aggregators[i]->aggregate(thread, &indices1d[0], INDEX_BLOCK_SIZE, offset);
                }
            }
            offset += (leftover < INDEX_BLOCK_SIZE) ? leftover : INDEX_BLOCK_SIZE;
            done = offset == length;
        }
    }
    std::vector<Binner *> binners;
    std::vector<uint64_t> strides;
    std::vector<uint64_t> shapes;
    uint64_t dimensions;
    size_t length1d;
};

} // namespace vaex

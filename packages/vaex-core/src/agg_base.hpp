
#include "agg.hpp"
#include "utils.hpp"
#include <condition_variable>
#include <mutex>

namespace vaex {

// base classes that make life easier

template <class GridType = double, class IndexType = default_index_type>
class AggregatorBase : public Aggregator {
  public:
    using index_type = IndexType;
    using grid_type = GridType;
    AggregatorBase(Grid<IndexType> *grid, int grids, int threads)
        : grid(grid), grid_used(grids, false), grids(grids), threads(threads), selection_mask_ptr(threads), selection_mask_size(threads), free_grids(grids) {
        grid_data = new grid_type[count()];
        if (grids != threads) {
            free_grids.resize(grids);
            for (int i = 0; i < grids; i++) {
                free_grids[i] = i;
            }
        }
    }
    virtual ~AggregatorBase() { delete[] grid_data; }

    virtual size_t count() { return grids * grid->length1d; }
    virtual size_t bytes_used() { return sizeof(grid_type) * count(); }
    void fill(grid_type fill_value, int grid) { std::fill(grid_data + this->grid->length1d * grid, grid_data + this->grid->length1d * (grid + 1), fill_value); }
    virtual void initial_fill(int grid) = 0;

    virtual void aggregate(int thread, default_index_type *indices1d, size_t length, uint64_t offset) {
        int grid = 0;
        if (grids == threads) {
            grid = thread;
        } else {
            grid = get();
        }
        if (!this->grid_used[grid]) {
            this->initial_fill(grid);
            this->grid_used[grid] = true;
        }
        this->aggregate(grid, thread, indices1d, length, offset);
        if (grids != threads) {
            put(grid);
        }
    };
    virtual void aggregate(int grid, int thread, default_index_type *indices1d, size_t length, uint64_t offset) = 0;

    void put(int grid_nr) {
        const std::lock_guard<std::mutex> lock(mutex);
        // printf("putting back: %i\n", grid_nr);
        free_grids.push_back(grid_nr);
        not_empty.notify_one();
    }
    int get() {
        // const std::lock_guard<std::mutex> lock(mutex);
        // not_empty.waia
        int result = 0;
        std::unique_lock<std::mutex> lock{mutex};
        if (free_grids.size() == 0) {
            // printf("all grids in use, wait for empty spot\n");
            do {
                not_empty.wait(lock);
                // we have exclusive lock on mutex, so we can pop
            } while (free_grids.size() == 0);
            // throw std::runtime_error("popping empty vector!");
            result = free_grids.back();
            // printf("waiting done, got grid: %i\n", result);
        } else {
            result = free_grids.back();
            // printf("directly got grid: %i\n", result);
        }
        free_grids.pop_back();
        return result;
    }

    void set_selection_mask(int thread, py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->selection_mask_ptr[thread] = (uint8_t *)info.ptr;
        this->selection_mask_size[thread] = info.shape[0];
    }
    void clear_selection_mask(int thread) {
        this->selection_mask_ptr[thread] = nullptr;
        this->selection_mask_size[thread] = 0;
    }

    Grid<IndexType> *grid;
    grid_type *grid_data; // no vector due to bool issues
    std::vector<bool> grid_used;
    int grids;
    int threads;
    std::vector<uint8_t *> selection_mask_ptr;
    std::vector<uint64_t> selection_mask_size;

    std::vector<int> free_grids;
    std::mutex mutex;
    std::condition_variable not_empty;
};


template <class C>  // make it compile for non primitive types
py::buffer_info agg_buffer_info(C* this_) {
    using grid_type = typename C::grid_type;
    std::vector<ssize_t> strides(this_->grid->dimensions + 1);
    std::vector<ssize_t> shapes(this_->grid->dimensions + 1);
    shapes[0] = this_->grids;
    std::copy(&this_->grid->shapes[0], &this_->grid->shapes[this_->grid->dimensions], &shapes[1]);
    std::transform(&this_->grid->strides[0], &this_->grid->strides[this_->grid->dimensions], &strides[1], [&](uint64_t x) { return x * sizeof(grid_type); });
    if (this_->grid->dimensions) {
        strides[0] = strides[1] * shapes[1];
    } else {
        strides[0] = sizeof(grid_type);
    }
    return py::buffer_info(&this_->grid_data[0],                              /* Pointer to buffer */
                            sizeof(grid_type),                          /* Size of one scalar */
                            py::format_descriptor<grid_type>::format(), /* Python struct-style format descriptor */
                            this_->grid->dimensions + 1,                       /* Number of dimensions */
                            shapes,                                     /* Buffer dimensions */
                            strides);
}

template <class GridType = double, class IndexType = default_index_type>
class AggregatorBaseNumpyData : public AggregatorBase<GridType, IndexType> {
  public:
    using Base = AggregatorBase<GridType, IndexType>;
    // set data is specific to subclasses, with regard to types
    AggregatorBaseNumpyData(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads), data_size(threads), data_mask_ptr(threads), data_mask_size(threads) {}
    virtual void set_data(int thread, py::buffer ar, size_t index) = 0;
    void set_data_mask(int thread, py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if ((size_t)thread >= this->data_mask_ptr.size()) {
            throw std::runtime_error("thread out of bound for data_mask_ptr");
        }
        if ((size_t)thread >= this->data_mask_size.size()) {
            throw std::runtime_error("thread out of bound for data_mask_size");
        }
        this->data_mask_ptr[thread] = (uint8_t *)info.ptr;
        this->data_mask_size[thread] = info.shape[0];
    }
    void clear_data_mask(int thread) {
        this->data_mask_ptr[thread] = nullptr;
        this->data_mask_size[thread] = 0;
    }
    std::vector<uint64_t> data_size;
    std::vector<uint8_t *> data_mask_ptr;
    std::vector<uint64_t> data_mask_size;
};

// takes 1 argument, of the DataType
template <class DataType = double, class GridType = DataType, class IndexType = default_index_type>
class AggregatorPrimitive : public AggregatorBaseNumpyData<GridType, IndexType> {
  public:
    using Base = AggregatorBaseNumpyData<GridType, IndexType>;
    using typename Base::index_type;
    using data_type = DataType;
    AggregatorPrimitive(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads), data_ptr(threads) {}

    virtual void set_data(int thread, py::buffer ar, size_t index) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if ((size_t)thread >= this->data_ptr.size()) {
            throw std::runtime_error("thread out of bound for data_ptr");
        }
        if ((size_t)thread >= this->data_size.size()) {
            throw std::runtime_error("thread out of bound for data_size");
        }
        this->data_ptr[thread] = (data_type *)info.ptr;
        this->data_size[thread] = info.shape[0];
    }
    std::vector<data_type *> data_ptr;
};

template <class GridType = uint64_t, class IndexType = default_index_type>
class AggBaseString : public AggregatorBase<GridType, IndexType> {
  public:
    using Base = AggregatorBase<GridType, IndexType>;
    using typename Base::index_type;
    using data_type = StringSequence;
    AggBaseString(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads), string_sequence(threads), data_mask_ptr(threads), data_mask_size(threads) {}
    ~AggBaseString() {}
    virtual void set_data(int thread, StringSequence *string_sequence, size_t index) { this->string_sequence[thread] = string_sequence; }
    void set_data_mask(int thread, py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr[thread] = (uint8_t *)info.ptr;
        this->data_mask_size[thread] = info.shape[0];
    }
    void clear_data_mask(int thread) {
        this->data_mask_ptr[thread] = nullptr;
        this->data_mask_size[thread] = 0;
    }
    // uint64_t data_size;
    std::vector<StringSequence *> string_sequence;
    std::vector<uint8_t *> data_mask_ptr;
    std::vector<uint64_t> data_mask_size;
};

template <class GridType = uint64_t, class IndexType = default_index_type>
class AggBaseObject : public AggregatorBase<IndexType> {
  public:
    using Base = AggregatorBase<IndexType>;
    using typename Base::index_type;
    using data_type = PyObject *;
    AggBaseObject(Grid<IndexType> *grid, int grids, int threads) : Base(grid, grids, threads), objects(grids), data_mask_ptr(threads), data_mask_size(threads), objects_size(threads) {}
    ~AggBaseObject() {}
    void set_data(int thread, py::buffer ar, size_t index) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        if ("O" != info.format) {
            std::string msg = "Expected object type";
            throw std::runtime_error(msg);
        }
        this->objects[thread] = (data_type *)info.ptr;
        this->objects_size[thread] = info.shape[0];
    }
    void clear_data_mask(int thread) {
        this->data_mask_ptr[thread] = nullptr;
        this->data_mask_size[thread] = 0;
    }
    void set_data_mask(int thread, py::buffer ar) {
        py::buffer_info info = ar.request();
        if (info.ndim != 1) {
            throw std::runtime_error("Expected a 1d array");
        }
        this->data_mask_ptr[thread] = (uint8_t *)info.ptr;
        this->data_mask_size[thread] = info.shape[0];
    }
    std::vector<data_type *> objects;
    std::vector<uint8_t *> data_mask_ptr;
    std::vector<uint64_t> data_mask_size;
    std::vector<uint64_t> objects_size;
};


template <class Agg, class Base, class Module>
void add_agg_binding_1arg(Module m, Base &base, const char *class_name) {
    py::class_<Agg>(m, class_name, base)
        .def(py::init<Grid<> *, int, int>(), py::keep_alive<1, 2>())
        .def_buffer(&agg_buffer_info<Agg>)
        .def("__sizeof__", &Agg::bytes_used)
        .def("set_data", &Agg::set_data)
        .def("clear_data_mask", &Agg::clear_data_mask)
        .def("set_data_mask", &Agg::set_data_mask)
        .def_property_readonly("grid", [](const Agg &agg) { return agg.grid; });
    ;
}

} // namespace vaex
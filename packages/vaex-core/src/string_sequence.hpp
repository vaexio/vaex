#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <nonstd/string_view.hpp>

typedef nonstd::string_view string_view;

namespace py = pybind11;

inline bool _is_null(uint8_t* null_bitmap, size_t i) {
    if(null_bitmap) {
        size_t byte_index = i / 8;
        size_t bit_index = (i % 8);
        return (null_bitmap[byte_index] & (1 << bit_index)) == 0;
    } else {
        return false;
    }
}

inline void _set_null(uint8_t* null_bitmap, size_t i) {
    size_t byte_index = i / 8;
    size_t bit_index = (i % 8);
    null_bitmap[byte_index] &= ~(1 << bit_index); // clears bit
}

inline void _clear_null(uint8_t* null_bitmap, size_t i) {
    size_t byte_index = i / 8;
    size_t bit_index = (i % 8);
    null_bitmap[byte_index] |= (1 << bit_index); // sets bit
}

class StringSequence {
    public:
    StringSequence(size_t length, uint8_t* null_bitmap=nullptr, int64_t null_offset=0) : length(length), null_bitmap(null_bitmap), null_offset(null_offset) {
    }
    virtual ~StringSequence() {
    }
    virtual size_t byte_size() const = 0;
    virtual bool is_null(size_t i) const {
        return _is_null(null_bitmap, i + null_offset);
    }
    virtual void set_null(size_t i) const {
        _set_null(null_bitmap, i);
    }
    virtual string_view view(size_t i) const = 0;
    virtual const std::string get(size_t i) const = 0;
    virtual std::unique_ptr<StringSequence> capitalize();
    virtual std::unique_ptr<StringSequence> lower();
    virtual std::unique_ptr<StringSequence> upper();
    virtual std::unique_ptr<StringSequence> lstrip(std::string chars);
    virtual std::unique_ptr<StringSequence> rstrip(std::string chars);
    virtual std::unique_ptr<StringSequence> strip(std::string chars);
    virtual StringSequence* concat(StringSequence* other);
    virtual py::object byte_length();
    virtual py::object len();
    py::object count(const std::string pattern, bool regex);
    py::object endswith(const std::string pattern);
    py::object startswith(const std::string pattern);
    py::object search(const std::string pattern, bool regex);
    py::object tolist();
    template<class T>
    StringSequence* lazy_index(py::array_t<T, py::array::c_style> indices);
    template<class T>
    StringSequence* index(py::array_t<T, py::array::c_style> indices);
    py::object get(size_t start, size_t end);
    py::object get_(size_t index) const;

    size_t length;
    uint8_t* null_bitmap;
    int64_t null_offset;
};


/* gives a lazy view on a StringSequence */
template<class T>
class StringSequenceLazyIndex : public StringSequence {
public:
    StringSequenceLazyIndex(StringSequence* string_sequence, T* indices, size_t length) : 
        StringSequence(length),
        string_sequence(string_sequence), indices(indices)
    {

    }
    virtual size_t byte_size() const {
        return string_sequence->byte_size();
    }
    virtual string_view view(size_t i) const {
        return string_sequence->view(indices[i]);
    }
    virtual const std::string get(size_t i) const {
        return string_sequence->get(indices[i]);
    };
    virtual bool is_null(size_t i) const {
        return string_sequence->is_null(indices[i]);
    }
    StringSequence* string_sequence;
    T* indices;
};
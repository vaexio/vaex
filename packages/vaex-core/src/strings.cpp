#include <iostream>
#include <string>
#include <algorithm>
#include <locale>
#include <regex>
#include <climits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include "superstring.hpp"

// #define VAEX_REGEX_USE_XPRESSIVE
#define VAEX_REGEX_USE_PCRE

// #ifdef VAEX_REGEX_USE_XPRESSIVE
#include <boost/xpressive/xpressive.hpp>
namespace xp = boost::xpressive;
// #endif

#ifdef VAEX_REGEX_USE_BOOST
#include <boost/regex.hpp>
#endif

#ifdef VAEX_REGEX_USE_PCRE
#include <pcrecpp.h>
#endif

#include "string_utils.hpp"
#include "unicode_utils.hpp"

namespace py = pybind11;

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

size_t utf8_slice_neg_index_to_byte_offset(string_view& source, int64_t end_negative_index) {
    int64_t len = str_len(source);
    int64_t positive_index = len + end_negative_index;
    const char *str = source.begin();
    const char *end = source.end();
    int64_t index = 0;
    while( (str < end) && (index < positive_index) ) {
        char current = *str;
        if(((unsigned char)current) < 0x80) {
            str += 1;
        } else if (((unsigned char)current) < 0xE0) {
            str += 2;
        } else if (((unsigned char)current) < 0xF0) {
            str += 3;
        } else if (((unsigned char)current) < 0xF8) {
            str += 4;
        }
        index += 1;
    }
    return str - source.begin(); // distance from start
}

size_t utf8_index_to_byte_offset(string_view& source, int64_t index) {
    // int64_t positive_index = len + end_negative_index;
    const char *str = source.begin();
    const char *end = source.end();
    while( (str < end) && (index > 0) ) {
        char current = *str;
        if(((unsigned char)current) < 0x80) {
            str += 1;
        } else if (((unsigned char)current) < 0xE0) {
            str += 2;
        } else if (((unsigned char)current) < 0xF0) {
            str += 3;
        } else if (((unsigned char)current) < 0xF8) {
            str += 4;
        }
        index--;
    }
    return str - source.begin(); // distance from start
}

class StringSequenceBase : public StringSequence {
    public:
    StringSequenceBase(size_t length, uint8_t* null_bitmap=nullptr, int64_t null_offset=0) : StringSequence(length, null_bitmap, null_offset) {
    }
    virtual ~StringSequenceBase() {
    }
    virtual void set_null(size_t i) const {
        _set_null(null_bitmap, i);
    }
    virtual StringSequenceBase* capitalize();
    virtual StringSequenceBase* concat(StringSequenceBase* other);
    virtual StringSequenceBase* concat2(std::string other);
    virtual StringSequenceBase* concat_reverse(std::string other);
    virtual StringSequenceBase* pad(int width, std::string fillchar, bool left, bool right);
    virtual StringSequenceBase* lower();
    virtual StringSequenceBase* upper();
    virtual StringSequenceBase* lstrip(std::string chars);
    virtual StringSequenceBase* rstrip(std::string chars);
    virtual StringSequenceBase* repeat(int64_t repeats);
    virtual StringSequenceBase* replace(std::string pattern, std::string replacement, int64_t n, int64_t flags, bool regex);
    virtual StringSequenceBase* strip(std::string chars);
    virtual StringSequenceBase* slice_string(int64_t start, int64_t stop);
    virtual StringSequenceBase* slice_string_end(int64_t start);
    virtual StringSequenceBase* title();
    virtual py::object isalnum();
    virtual py::object isalpha();
    virtual py::object isdigit();
    virtual py::object isspace();
    virtual py::object islower();
    virtual py::object isupper();
    // virtual py::object istitle();
    // virtual py::object isnumeric();
    // virtual py::object isdecimal();

    // virtual StringSequenceBase* rstrip();
    // virtual StringSequenceBase* strip();
    virtual py::object byte_length();
    virtual py::object len();
    py::object count(const std::string pattern, bool regex) {
        py::array_t<int64_t> counts(length);
        auto m = counts.mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            size_t pattern_length = pattern.size();
            if(regex) {
                // TODO: implement count using pcre
                // #if defined(VAEX_REGEX_USE_PCRE)
                //     pcrecpp::RE rex(pattern);
                // #elif defined(VAEX_REGEX_USE_XPRESSIVE)
                // #if defined(VAEX_REGEX_USE_XPRESSIVE)
                    xp::sregex rex = xp::sregex::compile(pattern);
                // #else
                    // std::regex rex(pattern);
                // #endif

                for(size_t i = 0; i < length; i++) {
                    // #if defined(VAEX_REGEX_USE_XPRESSIVE)
                        auto str = get(i); // TODO: can we use view(i)?
                        auto words_begin =  xp::sregex_iterator(str.begin(), str.end(), rex);
                        auto words_end = xp::sregex_iterator();
                        size_t count = std::distance(words_begin, words_end) ;
                    // #else
                    //     auto str = get(i);
                    //     auto words_begin =  std::sregex_iterator(str.begin(), str.end(), rex);
                    //     auto words_end = std::sregex_iterator();
                    //     size_t count = std::distance(words_begin, words_end) ;
                    // #endif
                    m(i) = count;
                }
            } else {
                for(size_t i = 0; i < length; i++) {
                    m(i) = 0;
                    auto str = view(i);
                    size_t offset = 0;
                    while((offset = str.find(pattern, offset)) != std::string::npos) {
                        offset += pattern_length;
                        m(i)++;
                    }
                }
            }
        }
        return std::move(counts);
    }
    py::object endswith(const std::string pattern) {
        py::array_t<bool> matches(length);
        auto m = matches.mutable_unchecked<1>();
        string_view pattern_view = pattern; // msvc doesn't like comparting string and string_view
        {
            py::gil_scoped_release release;
            size_t pattern_length = pattern.size();
            for(size_t i = 0; i < length; i++) {
                auto str = view(i);
                size_t string_length = str.length();
                int64_t skip = string_length - pattern_length;
                m(i) = ((skip >= 0) && str.substr(skip, pattern_length) == pattern_view);
            }
        }
        return std::move(matches);
    }
    py::object startswith(const std::string pattern) {
        py::array_t<bool> matches(length);
        auto m = matches.mutable_unchecked<1>();
        string_view pattern_view = pattern; // msvc doesn't like comparting string and string_view
        {
            py::gil_scoped_release release;
            size_t pattern_length = pattern.size();
            for(size_t i = 0; i < length; i++) {
                auto str = view(i);
                size_t string_length = str.length();
                m(i) = (string_length >= pattern_length) && str.substr(0, pattern_length) == pattern_view;
            }
        }
        return std::move(matches);
    }
    py::object find(const std::string pattern, int64_t start, int64_t end, bool till_end, bool left) {
        py::array_t<int64_t> indices(length);
        auto m = indices.mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            for(size_t i = 0; i < length; i++) {
                auto str = view(i);
                int64_t string_length = str.length();
                int64_t byte_start = utf8_index_to_byte_offset(str, start);
                int64_t byte_end = 0;
                if(till_end) {
                    byte_end = string_length;
                } else {
                    if(end > string_length) {
                        byte_end = string_length;
                    } else {
                        if(end < 0) {
                            byte_end = utf8_slice_neg_index_to_byte_offset(str, end);
                        } else {
                            byte_end = utf8_index_to_byte_offset(str, end);
                        }
                    }
                }
                int64_t find_index = -1;
                if( (byte_start < string_length) && (byte_end > byte_start) )  { // we need to start in the string
                    auto str_lookat = str.substr(byte_start, byte_end - byte_start);
                    if(left) {
                        find_index = str_lookat.find(pattern, 0);
                    } else {
                        find_index = str_lookat.rfind(pattern);
                    }
                }
                m(i) = find_index;
            }
        }
        return std::move(indices);
    }
    py::object search(const std::string pattern, bool regex) {
        py::array_t<bool> matches(length);
        auto m = matches.mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            if(regex) {
                #if defined(VAEX_REGEX_USE_PCRE)
                    pcrecpp::RE rex(pattern);
                #elif defined(VAEX_REGEX_USE_XPRESSIVE)
                    xp::sregex rex = xp::sregex::compile(pattern);
                #else
                    std::regex rex(pattern);
                #endif
                for(size_t i = 0; i < length; i++) {
                    #if defined(VAEX_REGEX_USE_PCRE)
                        std::string str = get(i);
                        bool match = rex.PartialMatch(str);
                    #elif defined(VAEX_REGEX_USE_XPRESSIVE)
                        std::string str = get(i);
                        bool match = xp::regex_search(str, rex);
                    #else
                        auto str = view(i);
                        bool match = regex_search(str, rex);
                    #endif
                    m(i) = match;
                }
            } else {
                for(size_t i = 0; i < length; i++) {
                    auto str = view(i);
                    m(i) = str.find(pattern) != std::string::npos;
                }
            }
        }
        return std::move(matches);
    }
    py::object match(const std::string pattern) {
         // same as search, but stricter (full regex should match)
        py::array_t<bool> matches(length);
        auto m = matches.mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            #if defined(VAEX_REGEX_USE_PCRE)
                pcrecpp::RE rex(pattern);
            #elif defined(VAEX_REGEX_USE_XPRESSIVE)
                xp::sregex rex = xp::sregex::compile(pattern);
            #else
                std::regex rex(pattern);
            #endif
            for(size_t i = 0; i < length; i++) {
                #if defined(VAEX_REGEX_USE_PCRE)
                    std::string str = get(i);
                    bool match = rex.FullMatch(str);
                #elif defined(VAEX_REGEX_USE_XPRESSIVE)
                    std::string str = get(i);
                    bool match = xp::regex_match(str, rex);
                #else
                    auto str = view(i);
                    bool match = regex_match(str, rex);
                #endif
                m(i) = match;
            }
        }
        return std::move(matches);
    }
    py::object equals(const std::string other) {
        py::array_t<bool> matches(length);
        auto m = matches.mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            for(size_t i = 0; i < length; i++) {
#if defined(_MSC_VER)
                auto str = get(i);
                bool match = str == other;
#else
                auto str = view(i);
                bool match = str == other;
#endif
                m(i) = match;
            }
        }
        return std::move(matches);
    }
    py::object equals2(const StringSequence* others) {
        py::array_t<bool> matches(length);
        if(length != others->length) {
            throw pybind11::index_error("equals should have equal string array lengths");
        }
        auto m = matches.mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            for(size_t i = 0; i < length; i++) {
                auto str = view(i);
                auto other = others->view(i);
                bool match = str == other;
                m(i) = match;
            }
        }
        return std::move(matches);
    }
    py::object tolist() {
        py::list l;
        for(size_t i = 0; i < length; i++) {
            l.append(get_(i));
        }
        return std::move(l);
    }
    template<class T>
    StringSequenceBase* lazy_index(py::array_t<T, py::array::c_style> indices);
    template<class T>
    StringSequenceBase* index(py::array_t<T, py::array::c_style> indices);

    py::object to_numpy() {
        npy_intp shape[1];
        shape[0] = length;
        PyObject* array = PyArray_SimpleNew(1, shape, NPY_OBJECT);
        PyArray_XDECREF((PyArrayObject*)array);
        PyObject **ptr = (PyObject**)PyArray_DATA((PyArrayObject*)array);
        for(size_t i = 0; i < length; i++) {
            string_view str = view(i);
            if(is_null(i)) {
                ptr[i] = Py_None;
                Py_INCREF(Py_None);
            } else {
                ptr[i] = PyUnicode_FromStringAndSize(str.begin(), str.length());;
            }
        }
        py::handle h = array;
        return py::reinterpret_steal<py::object>(h);
    }
    py::object get_(int64_t index) const {
        if((index < 0) || (index >= length)) {
            throw pybind11::index_error("index out of bounds");
        }
        if(is_null(index)) {
            return py::cast<py::none>(Py_None);
        } else {
            std::string str = get(index);
            return py::str(str);
        }
    }
};

/* gives a lazy view on a StringSequence */
template<class T>
class StringSequenceLazyIndex : public StringSequenceBase {
public:
    StringSequenceLazyIndex(StringSequenceBase* string_sequence, T* indices, size_t length) :
        StringSequenceBase(length),
        string_sequence(string_sequence), indices(indices)
    {

    }
    virtual size_t byte_size() const {
        return string_sequence->byte_size();
    }
    virtual string_view view(int64_t i) const {
        return string_sequence->view(indices[i]);
    }
    virtual const std::string get(int64_t i) const {
        return string_sequence->get(indices[i]);
    };
    virtual bool is_null(int64_t i) const {
        return string_sequence->is_null(indices[i]);
    }
    StringSequenceBase* string_sequence;
    T* indices;
};

template<class T>
StringSequenceBase* StringSequenceBase::lazy_index(py::array_t<T, py::array::c_style> indices) {
    py::buffer_info info = indices.request();
    if(info.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    return new StringSequenceLazyIndex<T>(this, (T*)info.ptr, info.shape[0]);
}

/* list(list(string)) structure */
class StringListList {
public:
    StringListList() {}
    StringListList(char *bytes, size_t byte_length, size_t length, size_t max_length2, size_t offset=0, uint8_t* null_bitmap=nullptr) 
       : bytes(bytes), byte_length(byte_length), indices1(0),
       length(length), offset(offset), max_length2(max_length2), null_bitmap(null_bitmap),
       _own_bytes(false), _own_indices(true) {
           indices1 = (int64_t*)malloc(sizeof(int64_t) * (length + 1));
           indices2 = (int64_t*)malloc(sizeof(int64_t) * (max_length2));
    }
    virtual ~StringListList() {
        if(_own_bytes) {
            free((void*)bytes);
        }
        if(_own_indices) {
            free((void*)indices1);
            free((void*)indices2);
        }
    }
    inline bool is_null(size_t i) const {
        return _is_null(null_bitmap, i);
    }
    void print() {
        for(size_t i = 0; i < length; i++) {
            _check1(i);
            int64_t substart = indices1[i] - offset;
            int64_t subend = indices1[i+1] - offset;
            size_t count = (subend - substart + 1)/2;
            std::cout << " >> count " << count << std::endl;
            for(size_t j = 0; j < count; j++) {
                int64_t start = indices2[substart + j*2];
                int64_t end = indices2[substart + j*2 + 1];
                std::cout << "  item " << j << " from " << start << " to " << end << std::endl;

            }
        }
    }
    void _check1(int64_t i) const {
        if( (i < 0) || (i > length) ) {
            throw std::runtime_error("string index out of bounds");
        }
        int64_t i1 = indices1[i] - offset;
        int64_t i2 = indices1[i+1] - offset;
        if( (i1 < 0) || (i1 > max_length2)) {
            throw std::runtime_error("out of bounds i1");
        }
        if( (i2 < 0) || (i2 > max_length2) ) {
            throw std::runtime_error("out of bounds i2");
        }
    }
    void _check2(int64_t i, int64_t j) const {
        _check1(i);
        int64_t substart = indices1[i] - offset;
        int64_t subend = indices1[i+1] - offset;
        size_t sublist_length = substart - subend;
        if( (j < 0) || (j > sublist_length) ) {
            throw std::runtime_error("string index2 out of bounds");
        }

        int64_t i1 = indices2[substart + j];
        int64_t i2 = indices2[substart + j + 1];
        if( (i1 < 0) || (i1 > byte_length)) {
            throw std::runtime_error("out of bounds i1");
        }
        if( (i2 < 0) || (i2 > byte_length) ) {
            throw std::runtime_error("out of bounds i2");
        }
    }
    virtual const std::string get(size_t i, size_t j) const {
        _check1(i);
        int64_t substart = indices1[i] - offset;
        int64_t start = indices2[substart + j*2];
        int64_t end = indices2[substart + j*2 + 1];
        int64_t count = end - start;
        return std::string(bytes + start, count);
    }
    virtual const py::object getlist(size_t i) const {
        if(is_null(i)) {
            return py::cast<py::none>(Py_None);
        } else {
            int64_t substart = indices1[i] - offset;
            int64_t subend = indices1[i+1] - offset;
            size_t count = (subend - substart + 1)/2;
            py::list l;
            for(size_t j = 0; j < count; j++) {
                l.append(std::string(get(i, j)));
            }
            return std::move(l);
        }
    }
    StringSequenceBase* join(std::string sep);
    py::list all() {
        py::list outer_list;
        for(size_t i = 0; i < length; i++) {
            outer_list.append(getlist(i));
        }
        return outer_list;
    }

public:
    char* bytes;
    size_t byte_length;
    int64_t* indices1;
    int64_t* indices2;
    size_t length;
    size_t offset;
    size_t max_length2;
    uint8_t* null_bitmap;
private:
    bool _own_bytes;
    bool _own_indices;
};

/* arrow like StringArray data structure */
template<class IC>
class StringList : public StringSequenceBase {
public:
    typedef IC index_type;
    typedef string_view value_value;

    StringList(char *bytes, size_t byte_length, index_type *indices, size_t length, size_t offset=0, uint8_t* null_bitmap=0, int64_t null_offset=0)
     : StringSequenceBase(length, null_bitmap, null_offset), bytes(bytes), byte_length(byte_length), indices(indices), offset(offset),
       _own_bytes(false), _own_indices(false), _own_null_bitmap(false) {
    }
    StringList(size_t byte_length, size_t string_count, index_type *indices, size_t offset=0, uint8_t* null_bitmap=0)
     : StringSequenceBase(string_count, null_bitmap), bytes(0), byte_length(byte_length), indices(indices), offset(offset),
     _own_bytes(true), _own_indices(false), _own_null_bitmap(false) {
         bytes = (char*)malloc(byte_length);
         _own_bytes = true;
    }
    StringList(size_t byte_length, size_t string_count, size_t offset=0, uint8_t* null_bitmap=0, int64_t null_offset=0)
     : StringSequenceBase(string_count, null_bitmap, null_offset), bytes(0), byte_length(byte_length), indices(0), offset(offset),
     _own_bytes(true), _own_indices(true), _own_null_bitmap(false) {
         bytes = (char*)malloc(byte_length);
         indices = (index_type*)malloc(sizeof(index_type) * (length + 1));
         _own_bytes = true;
    }
    virtual ~StringList() {
        if(_own_bytes) {
            free((void*)bytes);
        }
        if(_own_indices) {
            free((void*)indices);
        }
        if(_own_null_bitmap) {
            free((void*)null_bitmap);
        }
    }
    void add_null_bitmap() {
        _own_null_bitmap = true;
        size_t null_bitmap_length = (length + 7) / 8;
        null_bitmap = (unsigned char*)malloc(null_bitmap_length);
        memset(null_bitmap, 0xff, null_bitmap_length);
    }
    void ensure_null_bitmap() {
        if(null_bitmap == nullptr)
            add_null_bitmap();
    }
    void grow() {
        byte_length *= 2;
        bytes = (char*)realloc(bytes, byte_length);
    }
    virtual StringSequenceBase* capitalize();
    // virtual StringSequenceBase* lower();
    // virtual StringSequenceBase* upper();
    // a slice for when the indices are not filled yet
    StringList* slice_byte_offset(size_t i1, size_t i2, size_t byte_offset) {
        size_t byte_length = this->byte_length - byte_offset;
        return new StringList(bytes+byte_offset, byte_length, indices+i1, i2-i1, offset+byte_offset, null_bitmap, i1);
    }
    StringList* slice(size_t i1, size_t i2) {
        size_t byte_offset = indices[i1] - offset;
        size_t byte_length = indices[i2] - offset - byte_offset;
        return new StringList(bytes+byte_offset, byte_length, indices+i1, i2-i1, offset+byte_offset, null_bitmap, i1);
    }
    size_t fill_from(const StringSequence& from) {
        if(length < from.length) {
            throw std::runtime_error("index buffer too small");
        }
        size_t byte_offset = 0;
        for(size_t i = 0; i < from.length; i++) {
            indices[i] = byte_offset + offset;
            string_view str = from.view(i);
            size_t string_length = str.length();
            if(byte_offset + string_length > byte_length) {
                throw std::runtime_error("byte buffer too small");
            }
            std::copy(str.begin(), str.end(), bytes + byte_offset);
            if(from.is_null(i)) {
                if(!null_bitmap) {
                    throw std::runtime_error("source string sequence contains null values but target has no null bitmap allocated");
                } else {
                    _set_null(null_bitmap, i + null_offset);
                }
            } else {
                if(null_bitmap)
                    _clear_null(null_bitmap, i + null_offset);
            }
            byte_offset += string_length;
        }
        indices[length] = byte_offset + offset;
        return byte_offset;
    }
    virtual std::unique_ptr<StringListList> split(std::string pattern_) {
        py::gil_scoped_release release;

        const char* pattern = pattern_.c_str();
        size_t pattern_length = pattern_.length();

        size_t max_length_index2 = byte_length * 4;
        StringListList* sll = new StringListList(bytes, byte_length, length, max_length_index2, 0, null_bitmap);
        // worst case scenario, we split each string in single chars
        int64_t* offsets1 = sll->indices1;
        int64_t* offsets2 = sll->indices2;
        size_t index2 = 0;
        size_t index1 = 0;

        for(size_t i = 0; i < length; i++) {
            auto str_ = view(i);
            const char* str = str_.begin();
            auto string_length = str_.length();
            size_t string_offset = 0;
            int64_t bytes_offset = this->indices[i] - offset;
            // add index to first list item
            offsets1[index1] = index2;
            if(pattern_length == 0) { // whitespace splitter
                // strip leading whitespace
                while(string_length > 0 && ::isspace(str[string_offset])) {
                    string_offset++;
                    string_length--;
                }
                // and trailing
                if(string_length > 0) {
                    while(::isspace(str[string_offset+string_length-1]) && string_length > 0) {
                        string_length--;
                    }
                }
                if(string_length > 0) {
                    // add beginning of first string
                    while(string_length > 0) {
                        offsets2[index2] = bytes_offset + string_offset;
                        index2++;
                        // continue till we find whitespace or the end
                        while(!::isspace(str[string_offset]) && string_length > 0) {
                            string_offset++;
                            string_length--;
                        }
                        // and end of a string
                        offsets2[index2] = bytes_offset + string_offset;
                        index2++;
                        // ignore the whitespace
                        while(::isspace(str[string_offset]) && string_length > 0) {
                            string_offset++;
                            string_length--;
                        }
                    }
                }

            } else {
                // add beginning of first string
                offsets2[index2] = bytes_offset;
                index2++;
                while(string_offset + pattern_length <= string_length) {
                    if(pattern[0] == str[string_offset] && strncmp(pattern, str, pattern_length)) {
                        // add the end of the string found
                        offsets2[index2] = bytes_offset + string_offset;
                        index2++;
                        // add start of next string, skipping the pattern
                        // bytes_offset += pattern_length;
                        string_offset += pattern_length;
                        offsets2[index2] = bytes_offset + string_offset;
                        index2++;
                    } else {
                        string_offset++;
                    }
                }
            }
            index1++;
        }
        offsets2[index2] = byte_length;
        offsets1[index1] = index2;
        return std::unique_ptr<StringListList>(sll);
    };
    virtual size_t byte_size() const {
        return indices[length] - offset;
    };
    void print() {
        // std::cout << get();
    }
    void _check(size_t i) const {
        if( (i < 0) || (i > length) ) {
            throw std::runtime_error("string index out of bounds");
        }
        size_t i1 = indices[i] - offset;
        size_t i2 = indices[i+1] - offset;
        if( (i1 < 0) || (i1 > byte_length)) {
            throw std::runtime_error("out of bounds i1");
        }
        if( (i2 < 0) || (i2 > byte_length) ) {
            throw std::runtime_error("out of bounds i2");
        }

    }
    virtual string_view view() const {
        index_type start = indices[0] - offset;
        index_type end = indices[length] - offset;
        index_type count = end - start;
        return string_view(bytes + start, count);
    }
    virtual string_view view(int64_t i) const {
        // _check(i);
        size_t start = indices[i] - offset;
        size_t end = indices[i+1] - offset;
        size_t count = end - start;
        return string_view(bytes + start, count);
    }
    virtual const std::string get(int64_t i) const {
        // _check(i);
        size_t start = indices[i] - offset;
        size_t end = indices[i+1] - offset;
        size_t count = end - start;
        return std::string(bytes + start, count);
    }
public:
    char* bytes;
    size_t byte_length;
    index_type* indices;
    /* in order to make zero copy slices, the indices will need to be 'corrected' by the offset */
    size_t offset;
private:
    /* we can own some of the buffers */
    bool _own_bytes;
    bool _own_indices;
    bool _own_null_bitmap;
};

typedef StringList<int32_t> StringList32;
typedef StringList<int64_t> StringList64;

template<class C>
class utf8_appender {
public:
    C* container;
    int64_t bytes_left;
    char* str;
    utf8_appender(C* container) : container(container) {
        bytes_left = container->byte_length;
        str = container->bytes;
    }

    int64_t offset() const {
        return str - container->bytes;
    }



    void _check_buffer() {
        if(bytes_left < 0) {
            size_t used_bytes = str - container->bytes;
            int64_t missing_bytes = -bytes_left;
            size_t current_size = container->byte_length;
            while( int64_t(container->byte_length - current_size) < missing_bytes ) {
                container->grow();
            }
            bytes_left = container->byte_length - used_bytes;
            str = container->bytes + used_bytes;
        }
    }

    void append(char chr) {
        _check_buffer();
        bytes_left--;
        *str++ = chr;
    }

    void append(char32_t chr) {
        if(bytes_left >= 4) {
            if (chr < 0x80) {
                bytes_left--;
                *str++ = chr;
            }
            else if (chr < 0x800) {
                bytes_left -= 2;
                *str++ = 0xC0 + (chr >> 6);
                *str++ = 0x80 + (chr & 0x3F);
            }
            else if (chr < 0x10000) {
                bytes_left -= 3;
                *str++ = 0xE0 + (chr >> 12);
                *str++ = 0x80 + ((chr >> 6) & 0x3F);
                *str++ = 0x80 + (chr & 0x3F);
            }
            else if (chr < 0x200000) {
                bytes_left -= 4;
                *str++ = 0xF0 + (chr >> 18);
                *str++ = 0x80 + ((chr >> 12) & 0x3F);
                *str++ = 0x80 + ((chr >> 6) & 0x3F);
                *str++ = 0x80 + (chr & 0x3F);
            }
        } else {
            if (chr < 0x80) {
                bytes_left--;
                _check_buffer();
                *str++ = chr;
            }
            else if (chr < 0x800) {
                bytes_left -= 2;
                _check_buffer();
                *str++ = 0xC0 + (chr >> 6);
                *str++ = 0x80 + (chr & 0x3F);
            }
            else if (chr < 0x10000) {
                bytes_left -= 3;
                _check_buffer();
                *str++ = 0xE0 + (chr >> 12);
                *str++ = 0x80 + ((chr >> 6) & 0x3F);
                *str++ = 0x80 + (chr & 0x3F);
            }
            else if (chr < 0x200000) {
                bytes_left -= 4;
                _check_buffer();
                *str++ = 0xF0 + (chr >> 18);
                *str++ = 0x80 + ((chr >> 12) & 0x3F);
                *str++ = 0x80 + ((chr >> 6) & 0x3F);
                *str++ = 0x80 + (chr & 0x3F);
            }
        }
    }
};

template<class A, class U>
inline void utf8_transform(const string_view& source_view, char* target, A ascii_op, U unicode_op) {
    size_t length = source_view.length();
    const char *str = source_view.begin();
    // const char *end = source_view.end();
    size_t i = 0;
    while(i < length) {
        char current = *str;
        if(((unsigned char)current) < 0x80) {
            *target = ascii_op(current);
            i++;
            str++;
            target++;
        } else if (((unsigned char)current) < 0xE0) {
            char32_t c = unicode_op(utf8_decode(str));
            utf8_append(target, c);
            i += 2;
        } else if (((unsigned char)current) < 0xF0) {
            char32_t c = unicode_op(utf8_decode(str));
            utf8_append(target, c);
            i += 3;
        } else if (((unsigned char)current) < 0xF8) {
            char32_t c = unicode_op(utf8_decode(str));
            utf8_append(target, c);
            i += 4;
        }
    }
}



inline void copy(const string_view& source, char*& target) {
    std::copy(source.begin(), source.end(), target);
    target += source.length();
}



// templated implementation for _apply_seq
template<class StringList, class W>
StringSequenceBase* _apply_seq(StringSequenceBase* _this, W word_transform) {
    StringList* list = new StringList(_this->byte_size(), _this->length, 0, _this->null_bitmap, _this->null_offset);
    char* target = list->bytes;
    for(size_t i = 0; i < _this->length; i++) {
        list->indices[i] = target - list->bytes;
        string_view source = _this->view(i);
        word_transform(source, target);
        if(_this->is_null(i)) {
            list->ensure_null_bitmap();
            list->set_null(i);
        }
        // std::cout << " offset = " << list->indices[i] << std::endl;
        // target += length;
        // offset += length;
    }
    list->indices[_this->length] = target - list->bytes;
    return list;
}

// apply a function to each character, for each word
template<class W>
StringSequenceBase* _apply_seq(StringSequenceBase* _this, W word_transform) {
    py::gil_scoped_release release;
    if(_this->byte_size() > INT_MAX) {
        return _apply_seq<StringList64, W>(_this, word_transform);
    } else {
        return _apply_seq<StringList32, W>(_this, word_transform);
    }
}

template<class StringList, class W>
StringSequenceBase* _apply(StringList* _this, W word_transform) {
    py::gil_scoped_release release;
    StringList* list = new StringList(_this->byte_size(), _this->length, _this->offset, _this->null_bitmap);
    char* target = list->bytes;
    for(size_t i = 0; i < _this->length; i++) {
        target[_this->indices[i] - _this->offset] = target[_this->indices[i] - _this->offset];
        string_view source = _this->view(i);
        word_transform(source, target);
    }
    std::copy(_this->indices, _this->indices+_this->length+1, list->indices);
    return list;
}

// apply it to the whole buffer
template<class StringList, class W>
StringSequenceBase* _apply_all(StringList* _this, W word_transform) {
    py::gil_scoped_release release;
    StringList* list = new StringList(_this->byte_size(), _this->length, _this->offset, _this->null_bitmap);
    string_view source = _this->view();
    char* target = list->bytes;
    word_transform(source, target);
    std::copy(_this->indices, _this->indices+_this->length+1, list->indices);
    return list;
}

template<class T, class W>
py::array_t<T> _map(StringSequenceBase* _this, W word_op) {
    py::array_t<T> ar(_this->length);
    auto ar_unsafe = ar.template mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for(size_t i = 0; i < _this->length; i++) {
            string_view str = _this->view(i);
            ar_unsafe(i) = word_op(str);
        }
    }
    return py::array_t<T>(ar);
}

template<class T, class C>
py::array_t<T> _map_bool_all(StringSequenceBase* _this, C char_bool_op) {
    py::array_t<T> ar(_this->length);
    auto ar_unsafe = ar.template mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for(size_t i = 0; i < _this->length; i++) {
            string_view str = _this->view(i);
            bool all = true;
            if(str.length() == 0) {
                all = false;
            } else{
                all = std::all_of(str.begin(), str.end(), char_bool_op);
            }
            ar_unsafe(i) = all;
        }
    }
    return py::array_t<T>(ar);
}

bool always_true_ascii(char chr) { return true; }
bool always_true_unicode(char32_t chr) { return true; }

template<class T, class C, class U, class COR, class UOR>
py::array_t<T> _map_bool_all_utf8(StringSequenceBase* _this, C ascii_bool_op, U unicode_bool_op, COR ascii_op_or=always_true_ascii, UOR unicode_op_or=always_true_unicode) {
    py::array_t<T> ar(_this->length);
    auto ar_unsafe = ar.template mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for(size_t j = 0; j < _this->length; j++) {
            string_view str = _this->view(j);
            bool all = true;
            bool some = false;
            if(str.length() == 0) {
                all = false;
            } else {
                const char *i = str.begin();
                const char *end = str.end();
                while(i < end) {
                    char current = *i;
                    if(((unsigned char)current) < 0x80) {
                        some = some || ascii_op_or(current);
                        if(!ascii_bool_op(current)) {
                            all = false;
                            break;
                        }
                        i++;
                    } else if (((unsigned char)current) < 0xE0) {
                        char32_t c = utf8_decode(i);
                        some = some || unicode_op_or(c);
                        if(!unicode_bool_op(c)) {
                            all = false;
                            break;
                        }
                    } else if (((unsigned char)current) < 0xF0) {
                        char32_t c = utf8_decode(i);
                        some = some || unicode_op_or(c);
                        if(!unicode_bool_op(c)) {
                            all = false;
                            break;
                        }
                    } else if (((unsigned char)current) < 0xF8) {
                        char32_t c = utf8_decode(i);
                        some = some || unicode_op_or(c);
                        if(!unicode_bool_op(c)) {
                            all = false;
                            break;
                        }
                    }
                }
            }
            ar_unsafe(j) = all & some;
        }
    }
    return py::array_t<T>(ar);
}

class char_transformer_lower {
public:
    std::string chars;
    bool left, right;
    template<class A>
    void operator()(const string_view& source, A& appender) {
        const char* str = source.begin();
        const char* end = source.end();
        while(str < end) {
            char current = *str;
            if(((unsigned char)current) < 0x80) {
                char c = ::tolower(current);
                appender.append(c);
                str++;
            } else {
                char32_t c = ::char32_lowercase(utf8_decode(str));
                appender.append(c);
            }
        }
    }
};

class char_transformer_upper {
public:
    std::string chars;
    bool left, right;
    template<class A>
    void operator()(const string_view& source, A& appender) {
        const char* str = source.begin();
        const char* end = source.end();
        while(str < end) {
            char current = *str;
            if(((unsigned char)current) < 0x80) {
                char c = ::toupper(current);
                appender.append(c);
                str++;
            } else {
                char32_t c = ::char32_uppercase(utf8_decode(str));
                appender.append(c);
            }
        }
    }
};

// templated implementation for _apply2
template<class StringList, class W>
StringSequenceBase* _apply2(StringSequenceBase* _this, W word_transform) {
    StringList* list = new StringList(_this->byte_size(), _this->length, 0, _this->null_bitmap, _this->null_offset);
    utf8_appender<StringList> appender(list);
    for(size_t i = 0; i < _this->length; i++) {
        list->indices[i] = appender.offset();
        string_view source = _this->view(i);
        word_transform(source, appender);
        if(_this->is_null(i)) {
            list->ensure_null_bitmap();
            list->set_null(i);
        }
    }
    list->indices[_this->length] = appender.offset();
    return list;
}

// apply a function to each word
template<class W>
StringSequenceBase* _apply2(StringSequenceBase* _this, W word_transform) {
    py::gil_scoped_release release;
    // we are very conservative, it could be that a string
    // contains only unicode chars that uppercased become 2 bytes intead of 2
    // or maybe even 4 instead of 2, I assume never 2 or 3 instead of 1
    if(_this->byte_size() > (INT_MAX/2)) {
        return _apply2<StringList64, W>(_this, word_transform);
    } else {
        return _apply2<StringList32, W>(_this, word_transform);
    }
}

inline void lower(const string_view& source, char*& target) {
    utf8_transform(source, target, ::tolower, ::char32_lowercase);
    target += source.length();
}

StringSequenceBase* StringSequenceBase::lower() {
    return _apply2<>(this, char_transformer_lower());
}

inline void upper(const string_view& source, char*& target) {
    utf8_transform(source, target, ::toupper, ::char32_uppercase);
    target += source.length();
}

StringSequenceBase* StringSequenceBase::upper() {
    return _apply2<>(this, char_transformer_upper());
}

StringSequenceBase* StringSequenceBase::lstrip(std::string chars) {
    return _apply_seq<>(this, stripper(chars, true, false));
};

StringSequenceBase* StringSequenceBase::rstrip(std::string chars) {
    return _apply_seq<>(this, stripper(chars, false, true));
};

StringSequenceBase* StringSequenceBase::strip(std::string chars) {
    return _apply_seq<>(this, stripper(chars, true, true));
};

void capitalize(const string_view& source, char*& target) {
    size_t length = source.length();
    char* target_begin = target;
    if(length > 0) {
        lower(source, target);
        const char* str = source.begin();
        char32_t c = char32_uppercase(utf8_decode(str));
        utf8_append(target_begin, c);
        // target += length;
    }
}

StringSequenceBase* StringSequenceBase::capitalize() {
    return _apply_seq<>(this, ::capitalize);
}

template<class IC>
StringSequenceBase* StringList<IC>::capitalize() {
    return _apply<>(this, ::capitalize);
}

struct padder {
    size_t width;
    char fillchar;
    bool pad_left, pad_right;
    padder(size_t width, char fillchar, bool pad_left, bool pad_right) : width(width), fillchar(fillchar),
        pad_left(pad_left), pad_right(pad_right) {}
    void operator()(const string_view& source, char*& target) {
        size_t length = str_len(source);
        auto begin = source.begin();
        auto end = source.end();
        if(width > length) {
            int64_t left = 0, right = 0;
            if(pad_left & pad_right) {
                size_t margin = width - length;
                left = margin / 2 + (margin & width & 1);
                right = margin - left;
            } else if(pad_left) {
                left = width - length;
            } else if(pad_right) {
                right = width - length;
            }
            while(left) {
                (*target++) = fillchar;
                left--;
            }
            std::copy(begin, end, target);
            target += source.length();
            while(right) {
                (*target++) = fillchar;
                right--;
            }
        }
        else if(length) {
            std::copy(begin, end, target);
            target += source.length();
        }

    }    
};

StringSequenceBase* StringSequenceBase::pad(int width, std::string fillchar, bool left, bool right) {
    py::gil_scoped_release release;
    if(fillchar.length() != 1) {
        throw std::runtime_error("fillchar should be 1 character long (unicode not supported)");
    }
    char _fillchar = fillchar[0];
    auto p = padder(width, _fillchar, left, right);
    StringList64* sl = new StringList64(byte_size(), length);
    char* target = sl->bytes;
    size_t byte_offset;
    for(size_t i = 0; i < length; i++) {
        byte_offset = target - sl->bytes;
        sl->indices[i] = byte_offset;
        if(this->is_null(i)) {
            if(sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            string_view str = this->view(i);
            int64_t max_length = str.length() + width; // very safe choice
            while((byte_offset + max_length) > sl->byte_length) {
                sl->grow();
                target = sl->bytes + byte_offset;
            }
            p(str, target);
        }
    }
    byte_offset = target - sl->bytes;
    sl->indices[length] = byte_offset;
    return sl;
};


StringSequenceBase* StringSequenceBase::concat(StringSequenceBase* other) {
    py::gil_scoped_release release;
    if(other->length != this->length) {
        throw std::runtime_error("cannot concatenate unequal string sequences");
    }
    StringList64* sl = new StringList64(this->byte_size() + other->byte_size(), length);
    size_t byte_offset = 0;
    for(size_t i = 0; i < length; i++) {
        sl->indices[i] = byte_offset;
        if(this->is_null(i) || other->is_null(i)) {
            if(sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            string_view str1 = this->view(i);
            string_view str2 = other->view(i);
            std::copy(str1.begin(), str1.end(), sl->bytes + byte_offset);
            byte_offset += str1.length();
            std::copy(str2.begin(), str2.end(), sl->bytes + byte_offset);
            byte_offset += str2.length();
        }
    }
    sl->indices[length] = byte_offset;
    return sl;
}


StringSequenceBase* StringSequenceBase::concat2(std::string other) {
    py::gil_scoped_release release;
    size_t other_length = other.length();
    StringList64* sl = new StringList64(this->byte_size() + other_length * length, length);
    size_t byte_offset = 0;
    for(size_t i = 0; i < length; i++) {
        sl->indices[i] = byte_offset;
        if(this->is_null(i)) {
            if(sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            string_view str1 = this->view(i);
            std::copy(str1.begin(), str1.end(), sl->bytes + byte_offset);
            byte_offset += str1.length();
            std::copy(other.begin(), other.end(), sl->bytes + byte_offset);
            byte_offset += other_length;
        }
    }
    sl->indices[length] = byte_offset;
    return sl;
}

StringSequenceBase* StringSequenceBase::concat_reverse(std::string other) {
    py::gil_scoped_release release;
    size_t other_length = other.length();
    StringList64* sl = new StringList64(this->byte_size() + other_length * length, length);
    size_t byte_offset = 0;
    for(size_t i = 0; i < length; i++) {
        sl->indices[i] = byte_offset;
        if(this->is_null(i)) {
            if(sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            std::copy(other.begin(), other.end(), sl->bytes + byte_offset);
            byte_offset += other_length;
            string_view str1 = this->view(i);
            std::copy(str1.begin(), str1.end(), sl->bytes + byte_offset);
            byte_offset += str1.length();
        }
    }
    sl->indices[length] = byte_offset;
    return sl;
}

StringSequenceBase* StringSequenceBase::repeat(int64_t repeats) {
    py::gil_scoped_release release;
    StringList64* sl = new StringList64(this->byte_size() * repeats, length);
    size_t byte_offset = 0;
    for(size_t i = 0; i < length; i++) {
        sl->indices[i] = byte_offset;
        if(this->is_null(i)) {
            if(sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            string_view str = this->view(i);
            size_t string_length = str.length();
            for(int64_t i = 0; i < repeats; i++) {
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                byte_offset += string_length;
            }
        }
    }
    sl->indices[length] = byte_offset;
    return sl;
}


StringSequenceBase* StringSequenceBase::replace(std::string pattern, std::string replacement, int64_t n, int64_t flags, bool regex) {
    py::gil_scoped_release release;
    StringList64* sl = new StringList64(byte_size(), length);
    size_t byte_offset = 0;
    size_t pattern_length = pattern.length();
    size_t replacement_length = replacement.length();

    #if defined(VAEX_REGEX_USE_PCRE)
        pcrecpp::RE_Options opts;
        if(flags == 2) {
            opts.set_caseless(true);
        }
        pcrecpp::RE rex(pattern, opts);
    #elif defined(VAEX_REGEX_USE_XPRESSIVE)
        xp::regex_constants::syntax_option_type xp_flags = xp::regex_constants::ECMAScript;
        if(flags == 2) {
            xp_flags = xp_flags | xp::icase;
        }
        xp::sregex rex = xp::sregex::compile(pattern, xp_flags);
    #else
        std::regex_constants::syntax_option_type syntax_flags = std::regex_constants::ECMAScript;
        if(flags == 2) {
            syntax_flags |= std::regex_constants::icase;
        }
        std::regex rex(pattern, syntax_flags);
    #endif
    for(size_t i = 0; i < length; i++) {
        sl->indices[i] = byte_offset;
        if(this->is_null(i)) {
            if(sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            std::string str = this->get(i);
            size_t offset = 0;
            int count = 0;

            if(regex) {
                auto str = get(i);
                #if defined(VAEX_REGEX_USE_PCRE)
                    rex.GlobalReplace(replacement, &str);
                #elif defined(VAEX_REGEX_USE_XPRESSIVE)
                    str = xp::regex_replace(str, rex, replacement);
                #else
                    str = std::regex_replace(str, rex, replacement);
                #endif

                while(byte_offset + str.length() > sl->byte_length) {
                    sl->grow();
                }
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                byte_offset += str.length();
            } else {
                while( ((offset = str.find(pattern, offset)) != std::string::npos) && ((count < n) || ( n == -1)) ) {
                    // TODO: we can optimize this by writing out the substring and pattern, instead of calling replace
                    str = str.replace(offset, pattern_length, replacement);
                    offset += replacement_length;
                    count++;
                }
                while(byte_offset + str.length() > sl->byte_length) {
                    sl->grow();
                }
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                byte_offset += str.length();
            }
        }
    }
    sl->indices[length] = byte_offset;
    return sl;
}

struct slicer_copy {
    int64_t _start;
    int64_t _stop;
    bool till_end;
    slicer_copy(int64_t start, int64_t stop, bool till_end) : _start(start), _stop(stop), till_end(till_end) {}
    void operator()(const string_view& source, char*& target) {
        size_t byte_length = source.length();
        int64_t length = str_len(source);
        auto begin = source.begin();
        auto end = source.end();
        size_t seen = 0;
        int64_t start = _start;
        int64_t stop = _stop;
        if(start < 0) {
            start = std::max(int64_t(0), start + length);
        }
        int64_t skipped = 0;
        if(stop < 0) {
            stop = std::max(int64_t(0), stop + length);
        }
        if(start > stop && !till_end)
            return;
        if(start > 0) {
            while(start && (begin != end)) {
                char current = *begin;
                if(((unsigned char)current) < 0x80) {
                    begin++;
                } else if (((unsigned char)current) < 0xE0) {
                    begin +=2;
                } else if (((unsigned char)current) < 0xF0) {
                    begin +=3;
                } else if (((unsigned char)current) < 0xF8) {
                    begin +=4;
                }
                start--;
                skipped++;
            }
        }
        if(till_end) {
            byte_length = end - begin;
            std::copy(begin, end, target);
            target += byte_length;
        } else {
            size_t count = stop - start - skipped;
            while((begin != end) && (seen < count)) {
                char current = *begin;
                if(((unsigned char)current) < 0x80) {
                    *target = *begin;
                    begin++;
                    target++;
                } else if (((unsigned char)current) < 0xE0) {
                    *target = *begin;
                    begin++; target++;
                    *target = *begin;
                    begin++; target++;
                } else if (((unsigned char)current) < 0xF0) {
                    *target = *begin;
                    begin++; target++;
                    *target = *begin;
                    begin++; target++;
                    *target = *begin;
                    begin++; target++;
                } else if (((unsigned char)current) < 0xF8) {
                    *target = *begin;
                    begin++; target++;
                    *target = *begin;
                    begin++; target++;
                    *target = *begin;
                    begin++; target++;
                    *target = *begin;
                    begin++; target++;
                }
                seen++;
            }            
        }
    }
};


StringSequenceBase* StringSequenceBase::slice_string(int64_t start, int64_t stop) {
    return _apply_seq<>(this, slicer_copy(start, stop, false));
};
StringSequenceBase* StringSequenceBase::slice_string_end(int64_t start) {
    return _apply_seq<>(this, slicer_copy(start, -1, true));
};

void titlecase(const string_view& source, char*& target) {
    size_t length = source.length();
    char* i = target;
    char* end = target;// + length;
    if(length > 0) {
        lower(source, end);
        bool uppercase_next = true;
        while(i != end) {
            if(uppercase_next) {
                const char* str = i;
                char32_t c = char32_uppercase(utf8_decode(str));
                utf8_append(i, c);
                uppercase_next = false;
            }
            if(i != end) {
                if(::isspace(*i) || ::isdigit(*i)) {
                    uppercase_next = true;
                }
                i++;
            }
        }
        target = end;
    }
}


StringSequenceBase* StringSequenceBase::title() {
    return _apply_seq<>(this, ::titlecase);
}
/*
TODO: optimize many of these specific cases for StringList
template<class IC>
StringSequenceBase* StringList<IC>::title() {
    return _apply<>(this, ::titlecase);
}
*/

py::object StringSequenceBase::len() {
    return _map<int64_t>(this, ::str_len);
}

inline int64_t byte_length(const string_view& source) {
    return source.length();
}

py::object StringSequenceBase::byte_length() {
    return _map<int64_t>(this, ::byte_length);
}


py::object StringSequenceBase::isalnum() {
    return _map_bool_all_utf8<bool>(this, ::isalnum, char32_isalnum, always_true_ascii, always_true_unicode);
}
py::object StringSequenceBase::isalpha() {
    return _map_bool_all_utf8<bool>(this, ::isalpha, char32_isalpha, always_true_ascii, always_true_unicode);
}
py::object StringSequenceBase::isdigit() {
    return _map_bool_all<bool>(this, ::isdigit);
}
py::object StringSequenceBase::isspace() {
    return _map_bool_all<bool>(this, ::isspace);
}
// this will also allow spaces
bool case_isupper(char chr) {
    return chr == ::toupper(chr);
}
bool case_islower(char chr) {
    return chr == ::tolower(chr);
}
bool is_cased(char chr) {
    return ::tolower(chr) != ::toupper(chr);
}
bool is_cased_unicode(char32_t chr) {
    return char32_lowercase(chr) != char32_uppercase(chr);
}
py::object StringSequenceBase::islower() {
    return _map_bool_all_utf8<bool>(this, case_islower, utf8_islower, is_cased, is_cased_unicode);
}
py::object StringSequenceBase::isupper() {
    return _map_bool_all_utf8<bool>(this, case_isupper, utf8_isupper, is_cased, is_cased_unicode);
}
// py::object StringSequenceBase::istitle() {
//     return _map_bool_all<bool>(this, ::isalpha);
// }
// py::object StringSequenceBase::isnumeric() {
//     return _map_bool_all<bool>(this, ::isnumeric);
// }
// py::object StringSequenceBase::isdecimal() {
//     return _map_bool_all<bool>(this, ::isnumeric);
// }



const char* empty = "";

/* for a numpy array with dtype=object having strings */

class StringArray : public StringSequenceBase {
public:
    StringArray(PyObject** object_array, size_t length) : StringSequenceBase(length), _byte_size(0), _has_null(false) {
        #if PY_MAJOR_VERSION == 2
            utf8_objects = (PyObject**)malloc(length * sizeof(void*));
        #endif
        objects = (PyObject**)malloc(length * sizeof(void*));
        strings = (char**)malloc(length * sizeof(void*));
        sizes = (Py_ssize_t*)malloc(length * sizeof(Py_ssize_t));
        for(size_t i = 0; i < length; i++) {
            objects[i] = object_array[i];
            Py_IncRef(objects[i]);
            #if PY_MAJOR_VERSION == 3
                if(PyUnicode_CheckExact(object_array[i])) {
                    // python37 declares as const
                    strings[i] = (char*)PyUnicode_AsUTF8AndSize(object_array[i], &sizes[i]);
                } else {
                    strings[i] = nullptr;
                    _has_null = true;
                    sizes[i] = 0;
                }
            #else
                if(PyUnicode_CheckExact(object_array[i])) {
                    // if unicode, first convert to utf8
                    utf8_objects[i] = PyUnicode_AsUTF8String(object_array[i]);
                    sizes[i] = PyString_Size(utf8_objects[i]);
                    strings[i] = PyString_AsString(utf8_objects[i]);
                } else if(PyString_CheckExact(object_array[i])) {
                    // otherwise directly use
                    utf8_objects[i] = 0;
                    sizes[i] = PyString_Size(object_array[i]);
                    strings[i] = PyString_AsString(object_array[i]);
                } else {
                    strings[i] = nullptr;
                    _has_null = true;
                    utf8_objects[i] = nullptr;
                    sizes[i] = 0;
                }
            #endif
            _byte_size += sizes[i];
        }
    }
    ~StringArray() {
        free(strings);
        free(sizes);
        for(size_t i = 0; i < length; i++) {
            Py_XDECREF(objects[i]);
        }
        free(objects);
        
        #if PY_MAJOR_VERSION == 2
            for(size_t i = 0; i < length; i++) {
                Py_XDECREF(utf8_objects[i]);
            }
            free(utf8_objects);
        #endif
    }
    virtual size_t byte_size() const {
        return _byte_size;
    };
    virtual string_view view(int64_t i) const {
        if( (i < 0) || (i > length)) {
            throw std::runtime_error("index out of bounds");
        }
        if(strings[i] == 0) {
            return string_view(empty);
        }
        return string_view(strings[i], sizes[i]);
    }
    virtual const std::string get(int64_t i) const {
        if( (i < 0) || (i > length)) {
            throw std::runtime_error("index out of bounds");
        }
        if(strings[i] == 0) {
            return std::string(empty);
        }
        return std::string(strings[i], sizes[i]);
    }
    virtual bool has_null() const {
        return _has_null;
    }
    virtual bool is_null(int64_t i) const {
        return strings[i] == nullptr;
    }
    StringList64* to_arrow() {
        StringList64* sl = new StringList64(_byte_size, length);
        char* target = sl->bytes;
        for(size_t i = 0; i < length; i++) {
            sl->indices[i] = target - sl->bytes;
            if(is_null(i)) {
                sl->ensure_null_bitmap();
                sl->set_null(i);
            } else {
                auto str = view(i);
                ::copy(str, target);
            }
        }
        sl->indices[length] = target - sl->bytes;
        return sl;
    }
    #if PY_MAJOR_VERSION == 2
        PyObject** utf8_objects;
    #endif
    PyObject** objects;
    char** strings;
    Py_ssize_t* sizes;
private:
    size_t _byte_size;
    bool _has_null;
};

template<class T>
StringSequenceBase* StringSequenceBase::index(py::array_t<T, py::array::c_style> indices_) {
    py::buffer_info info = indices_.request();
    if(info.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    T* indices = (T*)info.ptr;
    size_t length = info.size;
    {
        py::gil_scoped_release release;
        StringList64* sl = new StringList64(length*2, length);
        size_t byte_offset = 0;
        for(size_t i = 0; i < length; i++) {
            T index = indices[i];
            std::string str = get(index);
            while(byte_offset + str.length() > sl->byte_length) {
                sl->grow();
            }
            std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
            if(is_null(index)) {
                if(sl->null_bitmap == nullptr)
                    sl->add_null_bitmap();
                sl->set_null(i);
            }
            sl->indices[i] = byte_offset;
            byte_offset += str.length();
        }
        sl->indices[length] = byte_offset;
        return sl;
    }
}

template<>
StringSequenceBase* StringSequenceBase::index<bool>(py::array_t<bool, py::array::c_style> mask_) {
    py::buffer_info info = mask_.request();
    if(info.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    bool* mask = (bool*)info.ptr;
    {
        py::gil_scoped_release release;
        size_t index_length = info.size;
        size_t length = 0;
        for(size_t i = 0; i < index_length; i++) {
            // std::cout << "bool mask  " << mask[i] << std::endl;
            if(mask[i])
                length++;
        }
        // std::cout << "bool mask length " << length << std::endl;
        StringList64* sl = new StringList64(length*2, length);
        size_t byte_offset = 0;
        int64_t index = 0;
        for(size_t i = 0; i < index_length; i++) {
            if(mask[i]) {
                std::string str = get(i);
                // std::cout << " ok " << i << " " << str << " " << index << std::endl;
                while(byte_offset + str.length() > sl->byte_length) {
                    sl->grow();
                }
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                if(is_null(i)) {
                    if(sl->null_bitmap == nullptr)
                        sl->add_null_bitmap();
                    sl->set_null(index);
                }
                sl->indices[index++] = byte_offset;
                byte_offset += str.length();
            }
        }
        sl->indices[length] = byte_offset;
        return sl;
    }
}

StringSequenceBase* StringListList::join(std::string sep) {
    py::gil_scoped_release release;
    StringList64* sl = new StringList64(1, length);
    char* target = sl->bytes;
    size_t byte_offset;
    for(size_t i = 0; i < length; i++) {
        byte_offset = target - sl->bytes;
        sl->indices[i] = byte_offset;
        if(this->is_null(i)) {
            if(sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            int64_t substart = indices1[i] - offset;
            int64_t subend = indices1[i+1] - offset;
            size_t count = (subend - substart + 1)/2;
            // string_view str = this->view(i);
            for(size_t j = 0; j < count; j++) {
                // l.append(std::string(get(i, j)));
                auto str = get(i, j);
                while((byte_offset + str.length()) > sl->byte_length) {
                    sl->grow();
                    target = sl->bytes + byte_offset;
                }
                copy(str, target);
                byte_offset = target - sl->bytes;
                if(j < (count - 1)) {

                    while((byte_offset + sep.length()) > sl->byte_length) {
                        sl->grow();
                        target = sl->bytes + byte_offset;
                    }
                    copy(sep, target);
                    byte_offset = target - sl->bytes;

                }
            }
        }
        sl->indices[length] = byte_offset;
    }
    byte_offset = target - sl->bytes;
    sl->indices[length] = byte_offset;
    return sl;
}

template<class StringList, class Base, class Module>
void add_string_list(Module m, Base& base, const char* class_name) {

    py::class_<StringList>(m, class_name, base)
        .def(py::init([](py::buffer bytes, py::array_t<typename StringList::index_type, py::array::c_style>& indices, size_t string_count, size_t offset) {
                py::buffer_info bytes_info = bytes.request();
                py::buffer_info indices_info = indices.request();
                if(bytes_info.ndim != 1) {
                    throw std::runtime_error("Expected a 1d byte buffer");
                }
                if(indices_info.ndim != 1) {
                    throw std::runtime_error("Expected a 1d indices buffer");
                }
                return new StringList((char*)bytes_info.ptr, bytes_info.shape[0],
                                   (typename StringList::index_type*)indices_info.ptr, string_count, offset
                                  );
            }), py::keep_alive<1, 2>(), py::keep_alive<1, 3>() // keep a reference to the ndarrays
        )
        // same ctor, duplicate code, cannot make null_bitmap accept None
        .def(py::init([](py::buffer bytes, py::array_t<typename StringList::index_type, py::array::c_style>& indices, size_t string_count, size_t offset,
                py::array_t<uint8_t, py::array::c_style> null_bitmap) {
                py::buffer_info bytes_info = bytes.request();
                py::buffer_info indices_info = indices.request();
                if(bytes_info.ndim != 1) {
                    throw std::runtime_error("Expected a 1d byte buffer");
                }
                if(indices_info.ndim != 1) {
                    throw std::runtime_error("Expected a 1d indices buffer");
                }
                uint8_t* null_bitmap_ptr = 0;
                if(null_bitmap) {
                    py::buffer_info null_bitmap_info = null_bitmap.request();
                    if(null_bitmap_info.ndim != 1) {
                        throw std::runtime_error("Expected a 1d indices buffer");
                    }
                    null_bitmap_ptr = (uint8_t*)null_bitmap_info.ptr;
                }
                return new StringList((char*)bytes_info.ptr, bytes_info.shape[0],
                                   (typename StringList::index_type*)indices_info.ptr, string_count, offset, null_bitmap_ptr
                                  );
            }), py::keep_alive<1, 2>(), py::keep_alive<1, 3>() // keep a reference to the ndarrays
        )
        .def("split", &StringList::split, py::keep_alive<0, 1>())
        .def("slice", &StringList::slice, py::keep_alive<0, 1>())
        .def("slice", &StringList::slice_byte_offset, py::keep_alive<0, 1>())
        .def("fill_from", &StringList::fill_from)
        // .def("get", (const std::string (StringList::*)(size_t))&StringList::get)
        // bug? we have to add this again
        // .def("get", (py::object (StringSequenceBase::*)(size_t, size_t))&StringSequenceBase::get, py::return_value_policy::take_ownership)
        .def_property_readonly("bytes", [](const StringList &sl) {
                auto capsule = py::capsule(&sl, [](void *v) {  });
                return py::array_t<char>(sl.byte_length, sl.bytes, capsule);
            }

        )
        .def_property_readonly("indices", [](const StringList &sl) {
                auto capsule = py::capsule(&sl, [](void *v) {  });
                return py::array_t<typename StringList::index_type>(sl.length+1, sl.indices, capsule);
            }
        )
        .def_property_readonly("null_bitmap", [](const StringList &sl) -> py::object {
                if(sl.null_bitmap) { // TODO: what if there is a lazy view
                    size_t length = (sl.length + 7) / 8;
                    return py::array_t<unsigned char>(length, sl.null_bitmap);
                } else  {
                    return py::cast<py::none>(Py_None);
                }
            }
        )
        .def_property_readonly("offset", [](const StringList &sl) {
                return sl.offset;
            }
        )
        .def_property_readonly("length", [](const StringList &sl) {
                return sl.length;
            }
        )
        // .def("__repr__",
        //     [](const StringList &sl) {
        //         return "<vaex.strings.StringList buffer='" + sl.get() + "'>";
        //     }
        // )
        ;
}

template<class T>
StringList64* to_string(py::array_t<T, py::array::c_style> values_) {
    size_t length = values_.size();
    auto values = values_. template unchecked<1>();
    if(values_.ndim() != 1) {
        throw std::runtime_error("Expected a 1d array");
    }
    {
        py::gil_scoped_release release;
        StringList64* sl = new StringList64(length*2, length);
        size_t byte_offset = 0;
        for(size_t i = 0; i < length; i++) {
            std::string str = std::to_string(values(i));
            while(byte_offset + str.length() > sl->byte_length) {
                sl->grow();
            }
            std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
            sl->indices[i] = byte_offset;
            byte_offset += str.length();
        }
        sl->indices[length] = byte_offset;
        return sl;
    }
}

template<class T>
StringList64* format(py::array_t<T, py::array::c_style> values_, const char* format) {
    size_t length = values_.size();
    auto values = values_. template unchecked<1>();
    if(values_.ndim() != 1) {
        throw std::runtime_error("Expected a 1d array");
    }
    {
        py::gil_scoped_release release;
        StringList64* sl = new StringList64(length*2, length);
        size_t byte_offset = 0;
        for(size_t i = 0; i < length; i++) {
            sl->indices[i] = byte_offset;
            bool done = false;
            int ret;
            while(!done) {
                int64_t bytes_left = sl->byte_length - byte_offset;
                ret = snprintf(sl->bytes + byte_offset, bytes_left, format, (T)values(i));
                if(ret < 0) {
                    throw std::runtime_error("Invalid format");
                } else if(ret < bytes_left) {
                    done = true;
                    byte_offset += strlen(sl->bytes + byte_offset);
                } else {
                    sl->grow();
                }
            }
        }
        sl->indices[length] = byte_offset;
        return sl;
    }
}

StringList64* format_string(StringSequence* values, const char* format) {
    size_t length = values->length;
    {
        py::gil_scoped_release release;
        StringList64* sl = new StringList64(length*2, length);
        size_t byte_offset = 0;
        for(size_t i = 0; i < length; i++) {
            sl->indices[i] = byte_offset;
            bool done = false;
            int ret;
            while(!done) {
                int64_t bytes_left = sl->byte_length - byte_offset;
                auto value = values->get(i);
                ret = snprintf(sl->bytes + byte_offset, bytes_left, format, value.c_str());
                if(ret < 0) {
                    throw std::runtime_error("Invalid format");
                } else if(ret < bytes_left) {
                    done = true;
                    byte_offset += strlen(sl->bytes + byte_offset);
                } else {
                    sl->grow();
                }
            }
        }
        sl->indices[length] = byte_offset;
        return sl;
    }
}

PYBIND11_MODULE(superstrings, m) {
    _import_array();
    m.doc() = "fast operations on string sequences";
    py::class_<StringSequence> string_sequence(m, "StringSequence");
    py::class_<StringSequenceBase> string_sequence_base(m, "StringSequenceBase", string_sequence);
    string_sequence_base
        .def("to_numpy", &StringSequenceBase::to_numpy, py::return_value_policy::take_ownership)
        .def("lazy_index", &StringSequenceBase::lazy_index<int32_t>, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("lazy_index", &StringSequenceBase::lazy_index<int64_t>, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("lazy_index", &StringSequenceBase::lazy_index<uint32_t>, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("lazy_index", &StringSequenceBase::lazy_index<uint64_t>, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("index", &StringSequenceBase::index<bool>)
        .def("index", &StringSequenceBase::index<int32_t>)
        .def("index", &StringSequenceBase::index<int64_t>)
        .def("index", &StringSequenceBase::index<uint32_t>)
        .def("index", &StringSequenceBase::index<uint64_t>)
        .def("tolist", &StringSequenceBase::tolist)
        .def("capitalize", &StringSequenceBase::capitalize, py::keep_alive<0, 1>())
        .def("concat", &StringSequenceBase::concat)
        .def("concat_reverse", &StringSequenceBase::concat_reverse)
        .def("concat", &StringSequenceBase::concat2)
        .def("pad", &StringSequenceBase::pad)
        .def("search", &StringSequenceBase::search, "Tests if strings contains pattern", py::arg("pattern"), py::arg("regex"))//, py::call_guard<py::gil_scoped_release>())
        .def("count", &StringSequenceBase::count, "Count occurrences of pattern", py::arg("pattern"), py::arg("regex"))
        .def("upper", &StringSequenceBase::upper)
        .def("endswith", &StringSequenceBase::endswith)
        .def("find", &StringSequenceBase::find)
        .def("lower", &StringSequenceBase::lower)
        .def("match", &StringSequenceBase::match, "Tests if strings matches regex", py::arg("pattern"))
        .def("equals", &StringSequenceBase::equals, "Tests if strings are equal")
        .def("equals", &StringSequenceBase::equals2, "Tests if strings are equal")
        .def("lstrip", &StringSequenceBase::lstrip)
        .def("rstrip", &StringSequenceBase::rstrip)
        .def("repeat", &StringSequenceBase::repeat)
        .def("replace", &StringSequenceBase::replace)
        .def("startswith", &StringSequenceBase::startswith)
        .def("strip", &StringSequenceBase::strip)
        .def("slice_string", &StringSequenceBase::slice_string)
        .def("slice_string_end", &StringSequenceBase::slice_string_end)
        .def("title", &StringSequenceBase::title)
        .def("isalnum", &StringSequenceBase::isalnum)
        .def("isalpha", &StringSequenceBase::isalpha)
        .def("isdigit", &StringSequenceBase::isdigit)
        .def("isspace", &StringSequenceBase::isspace)
        .def("islower", &StringSequenceBase::islower)
        .def("isupper", &StringSequenceBase::isupper)
        // .def("istitle", &StringSequenceBase::istitle)
        // .def("isnumeric", &StringSequenceBase::isnumeric)
        // .def("isdecimal", &StringSequenceBase::isdecimal)
        .def("len", &StringSequenceBase::len)
        .def("byte_length", &StringSequenceBase::byte_length)
        .def("get", &StringSequenceBase::get_)
        .def("mask", [](const StringSequence &sl) -> py::object {
                if(sl.null_bitmap) { // TODO: what if there is a lazy view
                    auto ar = py::array_t<bool>(sl.length);
                    auto ar_unsafe = ar.mutable_unchecked<1>();
                    {
                        py::gil_scoped_release release;
                        for(size_t i = 0; i < sl.length; i++) {
                            ar_unsafe(i) = sl.is_null(i);
                        }
                    }
                    return std::move(ar);
                } else  {
                    return py::cast<py::none>(Py_None);
                }
            }
        )
    ;
    py::class_<StringListList>(m, "StringListList")
        .def("all", &StringListList::all)
        .def("get", &StringListList::get)
        .def("join", &StringListList::join)
        .def("get", &StringListList::getlist)
        .def("print", &StringListList::print)
        .def("__len__", [](const StringListList &obj) { return obj.length; })
    ;
    add_string_list<StringList32>(m, string_sequence_base, "StringList32");
    add_string_list<StringList64>(m, string_sequence_base, "StringList64");
    py::class_<StringArray>(m, "StringArray", string_sequence_base)
        .def(py::init([](py::buffer string_array) {
                py::buffer_info info = string_array.request();
                if(info.ndim != 1) {
                    throw std::runtime_error("Expected a 1d byte buffer");
                }
                if(info.format != "O") {
                    throw std::runtime_error("Expected an object array");
                }
                // std::cout << info.format << " format" << std::endl;
                return std::unique_ptr<StringArray>(
                    new StringArray((PyObject**)info.ptr, info.shape[0]));
            }) // no need to keep a reference to the ndarrays
        )
        .def("to_arrow", &StringArray::to_arrow) // nothing to keep alive, all a copy
        // .def("get", &StringArray::get_)
        // .def("get", (const std::string (StringArray::*)(int64_t))&StringArray::get)
        // bug? we have to add this again
        // .def("get", (py::object (StringSequenceBase::*)(size_t, size_t))&StringSequenceBase::get, py::return_value_policy::take_ownership)
        .def("mask", [](const StringSequence &sl) -> py::object {
                bool has_null = false;
                auto ar = py::array_t<bool>(sl.length);
                auto ar_unsafe = ar.mutable_unchecked<1>();
                {
                    py::gil_scoped_release release;
                    for(size_t i = 0; i < sl.length; i++) {
                        ar_unsafe(i) = sl.is_null(i);
                        has_null |= sl.is_null(i);
                    }
                }
                return std::move(ar);
                // if(has_null) {
                //     return std::move(ar);
                // } else  {
                //     return py::cast<py::none>(Py_None);
                // }
            }
        )
        ;
    m.def("to_string", &to_string<float>);
    m.def("to_string", &to_string<double>);
    m.def("to_string", &to_string<int64_t>);
    m.def("to_string", &to_string<int32_t>);
    m.def("to_string", &to_string<int16_t>);
    m.def("to_string", &to_string<int8_t>);
    m.def("to_string", &to_string<uint64_t>);
    m.def("to_string", &to_string<uint32_t>);
    m.def("to_string", &to_string<uint16_t>);
    m.def("to_string", &to_string<uint8_t>);
    m.def("to_string", &to_string<bool>);
    m.def("format", &format<float>);
    m.def("format", &format<double>);
    m.def("format", &format<int64_t>);
    m.def("format", &format<int32_t>);
    m.def("format", &format<int16_t>);
    m.def("format", &format<int8_t>);
    m.def("format", &format<uint64_t>);
    m.def("format", &format<uint32_t>);
    m.def("format", &format<uint16_t>);
    m.def("format", &format<uint8_t>);
    m.def("format", &format<bool>);
    m.def("format", &format_string);
}

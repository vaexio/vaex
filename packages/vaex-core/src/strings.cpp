#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "superstring.hpp"
#include <algorithm>
#include <climits>
#include <iostream>
#include <locale>
#include <string>

#include <regex>

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

py::object StringSequenceBase::count(const std::string pattern, bool regex) {
    py::array_t<int64_t> counts(length);
    auto m = counts.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        size_t pattern_length = pattern.size();
        if (regex) {
            // TODO: implement count using pcre
            // #if defined(VAEX_REGEX_USE_PCRE)
            //     pcrecpp::RE rex(pattern);
            // #elif defined(VAEX_REGEX_USE_XPRESSIVE)
            // #if defined(VAEX_REGEX_USE_XPRESSIVE)
            xp::sregex rex = xp::sregex::compile(pattern);
            // #else
            // std::regex rex(pattern);
            // #endif

            for (size_t i = 0; i < length; i++) {
                // #if defined(VAEX_REGEX_USE_XPRESSIVE)
                auto str = get(i); // TODO: can we use view(i)?
                auto words_begin = xp::sregex_iterator(str.begin(), str.end(), rex);
                auto words_end = xp::sregex_iterator();
                size_t count = std::distance(words_begin, words_end);
                // #else
                //     auto str = get(i);
                //     auto words_begin =  std::sregex_iterator(str.begin(), str.end(), rex);
                //     auto words_end = std::sregex_iterator();
                //     size_t count = std::distance(words_begin, words_end) ;
                // #endif
                m(i) = count;
            }
        } else {
            for (size_t i = 0; i < length; i++) {
                m(i) = 0;
                auto str = view(i);
                size_t offset = 0;
                while ((offset = str.find(pattern, offset)) != std::string::npos) {
                    offset += pattern_length;
                    m(i)++;
                }
            }
        }
    }
    return std::move(counts);
}

py::object StringSequenceBase::search(const std::string pattern, bool regex) {
    py::array_t<bool> matches(length);
    auto m = matches.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        if (regex) {
#if defined(VAEX_REGEX_USE_PCRE)
            pcrecpp::RE rex(pattern);
#elif defined(VAEX_REGEX_USE_XPRESSIVE)
            xp::sregex rex = xp::sregex::compile(pattern);
#else
            std::regex rex(pattern);
#endif
            for (size_t i = 0; i < length; i++) {
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
            for (size_t i = 0; i < length; i++) {
                auto str = view(i);
                m(i) = str.find(pattern) != std::string::npos;
            }
        }
    }
    return std::move(matches);
}

py::object StringSequenceBase::match(const std::string pattern) {
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
        for (size_t i = 0; i < length; i++) {
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

// inline virtual StringSequenceBase* replace(std::string pattern, std::string replacement, int64_t n, int64_t flags, bool regex);
StringSequenceBase *StringSequenceBase_replace(StringSequenceBase *this_, std::string pattern, std::string replacement, int64_t n, int64_t flags, bool regex) {
    py::gil_scoped_release release;
    StringList64 *sl = new StringList64(this_->byte_size(), this_->length);
    size_t byte_offset = 0;
    size_t pattern_length = pattern.length();
    size_t replacement_length = replacement.length();

#if defined(VAEX_REGEX_USE_PCRE)
    pcrecpp::RE_Options opts;
    if (flags == 2) {
        opts.set_caseless(true);
    }
    pcrecpp::RE rex(pattern, opts);
#elif defined(VAEX_REGEX_USE_XPRESSIVE)
    xp::regex_constants::syntax_option_type xp_flags = xp::regex_constants::ECMAScript;
    if (flags == 2) {
        xp_flags = xp_flags | xp::icase;
    }
    xp::sregex rex = xp::sregex::compile(pattern, xp_flags);
#else
    std::regex_constants::syntax_option_type syntax_flags = std::regex_constants::ECMAScript;
    if (flags == 2) {
        syntax_flags |= std::regex_constants::icase;
    }
    std::regex rex(pattern, syntax_flags);
#endif
    for (size_t i = 0; i < this_->length; i++) {
        sl->indices[i] = byte_offset;
        if (this_->is_null(i)) {
            if (sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            std::string str = this_->get(i);
            size_t offset = 0;
            int count = 0;

            if (regex) {
                auto str = this_->get(i);
#if defined(VAEX_REGEX_USE_PCRE)
                rex.GlobalReplace(replacement, &str);
#elif defined(VAEX_REGEX_USE_XPRESSIVE)
                str = xp::regex_replace(str, rex, replacement);
#else
                str = std::regex_replace(str, rex, replacement);
#endif

                while (byte_offset + str.length() > sl->byte_length) {
                    sl->grow();
                }
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                byte_offset += str.length();
            } else {
                while (((offset = str.find(pattern, offset)) != std::string::npos) && ((count < n) || (n == -1))) {
                    // TODO: we can optimize this by writing out the substring and pattern, instead of calling replace
                    str = str.replace(offset, pattern_length, replacement);
                    offset += replacement_length;
                    count++;
                }
                while (byte_offset + str.length() > sl->byte_length) {
                    sl->grow();
                }
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                byte_offset += str.length();
            }
        }
    }
    sl->indices[this_->length] = byte_offset;
    return sl;
}

const char *empty = "";

/* for a numpy array with dtype=object having strings */

class StringArray : public StringSequenceBase {
  public:
    StringArray(PyObject **object_array, size_t length, uint8_t *byte_mask = nullptr) : StringSequenceBase(length), _byte_size(0), _has_null(false) {
#if PY_MAJOR_VERSION == 2
        utf8_objects = (PyObject **)malloc(length * sizeof(void *));
#endif
        objects = (PyObject **)malloc(length * sizeof(void *));
        strings = (char **)malloc(length * sizeof(void *));
        sizes = (Py_ssize_t *)malloc(length * sizeof(Py_ssize_t));
        for (size_t i = 0; i < length; i++) {
            objects[i] = object_array[i];
            Py_IncRef(objects[i]);
#if PY_MAJOR_VERSION == 3
            if (PyUnicode_CheckExact(object_array[i]) && ((byte_mask == nullptr) || (byte_mask[i] == 0))) {
                // python37 declares as const
                strings[i] = (char *)PyUnicode_AsUTF8AndSize(object_array[i], &sizes[i]);
            } else {
                strings[i] = nullptr;
                _has_null = true;
                sizes[i] = 0;
            }
#else
            if (PyUnicode_CheckExact(object_array[i]) && ((byte_mask == nullptr) || (byte_mask[i] == 0))) {
                // if unicode, first convert to utf8
                utf8_objects[i] = PyUnicode_AsUTF8String(object_array[i]);
                sizes[i] = PyString_Size(utf8_objects[i]);
                strings[i] = PyString_AsString(utf8_objects[i]);
            } else if (PyString_CheckExact(object_array[i]) && ((byte_mask == nullptr) || (byte_mask[i] == 0))) {
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
        for (size_t i = 0; i < length; i++) {
            Py_XDECREF(objects[i]);
        }
        free(objects);

#if PY_MAJOR_VERSION == 2
        for (size_t i = 0; i < length; i++) {
            Py_XDECREF(utf8_objects[i]);
        }
        free(utf8_objects);
#endif
    }
    virtual size_t byte_size() const { return _byte_size; };
    virtual string_view view(int64_t i) const {
        if ((i < 0) || (i > length)) {
            throw std::runtime_error("index out of bounds");
        }
        if (strings[i] == 0) {
            return string_view(empty);
        }
        return string_view(strings[i], sizes[i]);
    }
    virtual const std::string get(int64_t i) const {
        if ((i < 0) || (i > length)) {
            throw std::runtime_error("index out of bounds");
        }
        if (strings[i] == 0) {
            return std::string(empty);
        }
        return std::string(strings[i], sizes[i]);
    }
    virtual bool has_null() const { return _has_null; }
    virtual bool is_null(int64_t i) const { return strings[i] == nullptr; }
    StringList64 *to_arrow() {
        StringList64 *sl = new StringList64(_byte_size, length);
        char *target = sl->bytes;
        for (size_t i = 0; i < length; i++) {
            sl->indices[i] = target - sl->bytes;
            if (is_null(i)) {
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
    PyObject **utf8_objects;
#endif
    PyObject **objects;
    char **strings;
    Py_ssize_t *sizes;

  private:
    size_t _byte_size;
    bool _has_null;
};

template <class T>
StringSequenceBase *StringSequenceBase::index(py::array_t<T, py::array::c_style> indices_) {
    py::buffer_info info = indices_.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    T *indices = (T *)info.ptr;
    size_t length = info.size;
    {
        py::gil_scoped_release release;
        StringList64 *sl = new StringList64(length * 2, length);
        size_t byte_offset = 0;
        for (size_t i = 0; i < length; i++) {
            T index = indices[i];
            std::string str = get(index);
            while (byte_offset + str.length() > sl->byte_length) {
                sl->grow();
            }
            std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
            if (is_null(index)) {
                if (sl->null_bitmap == nullptr)
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

template <class T>
StringSequenceBase *StringSequenceBase::index_masked(py::array_t<T, py::array::c_style> indices_, py::array_t<bool, py::array::c_style> mask_) {
    py::buffer_info info = indices_.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    T *indices = (T *)info.ptr;
    size_t length = info.size;

    py::buffer_info info_mask = mask_.request();
    if (info_mask.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    bool *mask = (bool *)info_mask.ptr;
    if (info_mask.size != info.size) {
        throw std::runtime_error("Indices and mask are of unequal length");
    }

    {
        py::gil_scoped_release release;
        StringList64 *sl = new StringList64(length * 2, length);
        size_t byte_offset = 0;
        for (size_t i = 0; i < length; i++) {
            sl->indices[i] = byte_offset;
            T index = indices[i];
            if ((mask[i] == 1) || is_null(index)) {
                if (sl->null_bitmap == nullptr)
                    sl->add_null_bitmap();
                sl->set_null(i);
            } else {
                std::string str = get(index);
                while (byte_offset + str.length() > sl->byte_length) {
                    sl->grow();
                }
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                byte_offset += str.length();
            }
        }
        sl->indices[length] = byte_offset;
        return sl;
    }
}

template <>
StringSequenceBase *StringSequenceBase::index<bool>(py::array_t<bool, py::array::c_style> mask_) {
    py::buffer_info info = mask_.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    bool *mask = (bool *)info.ptr;
    {
        py::gil_scoped_release release;
        size_t index_length = info.size;
        size_t length = 0;
        for (size_t i = 0; i < index_length; i++) {
            // std::cout << "bool mask  " << mask[i] << std::endl;
            if (mask[i])
                length++;
        }
        // std::cout << "bool mask length " << length << std::endl;
        StringList64 *sl = new StringList64(length * 2, length);
        size_t byte_offset = 0;
        int64_t index = 0;
        for (size_t i = 0; i < index_length; i++) {
            if (mask[i]) {
                std::string str = get(i);
                // std::cout << " ok " << i << " " << str << " " << index << std::endl;
                while (byte_offset + str.length() > sl->byte_length) {
                    sl->grow();
                }
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                if (is_null(i)) {
                    if (sl->null_bitmap == nullptr)
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

StringSequenceBase *StringListList::join(std::string sep) {
    py::gil_scoped_release release;
    StringList64 *sl = new StringList64(1, length);
    char *target = sl->bytes;
    size_t byte_offset;
    for (size_t i = 0; i < length; i++) {
        byte_offset = target - sl->bytes;
        sl->indices[i] = byte_offset;
        if (this->is_null(i)) {
            if (sl->null_bitmap == nullptr)
                sl->add_null_bitmap();
            sl->set_null(i);
        } else {
            int64_t substart = indices1[i] - offset;
            int64_t subend = indices1[i + 1] - offset;
            size_t count = (subend - substart + 1) / 2;
            // string_view str = this->view(i);
            for (size_t j = 0; j < count; j++) {
                // l.append(std::string(get(i, j)));
                auto str = get(i, j);
                while ((byte_offset + str.length()) > sl->byte_length) {
                    sl->grow();
                    target = sl->bytes + byte_offset;
                }
                copy(str, target);
                byte_offset = target - sl->bytes;
                if (j < (count - 1)) {

                    while ((byte_offset + sep.length()) > sl->byte_length) {
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

template <class T>
T *join(std::string sep, py::array_t<typename T::index_type, py::array::c_style> offsets_list, T *input, int64_t offset = 0) {
    py::gil_scoped_release release;
    int64_t list_length = offsets_list.size() - 1;
    auto offsets = offsets_list.template mutable_unchecked<1>();
    T *sl = new T(1, list_length);
    char *target = sl->bytes;
    size_t byte_offset;
    for (int64_t i = 0; i < list_length; i++) {
        byte_offset = target - sl->bytes;
        sl->indices[i] = byte_offset;
        int64_t i1 = offsets[i] - offset;
        int64_t i2 = offsets[i + 1] - offset;
        size_t count = i2 - i1;
        for (size_t j = 0; j < count; j++) {
            auto str = input->get(i1 + j);
            // make sure the buffer is large enough
            while ((byte_offset + str.length()) > sl->byte_length) {
                sl->grow();
                target = sl->bytes + byte_offset;
            }
            copy(str, target);
            byte_offset = target - sl->bytes;
            // copy separator
            if (j < (count - 1)) {

                while ((byte_offset + sep.length()) > sl->byte_length) {
                    sl->grow();
                    target = sl->bytes + byte_offset;
                }
                copy(sep, target);
                byte_offset = target - sl->bytes;
            }
        }
    }
    byte_offset = target - sl->bytes;
    sl->indices[list_length] = byte_offset;
    return sl;
}

template <class StringList, class Base, class Module>
void add_string_list(Module m, Base &base, const char *class_name) {

    py::class_<StringList, std::shared_ptr<StringList>>(m, class_name, base)
        .def(py::init([](py::buffer bytes, py::buffer indices, size_t string_count, size_t offset) {
                 py::buffer_info bytes_info = bytes.request();
                 py::buffer_info indices_info = indices.request();
                 if (bytes_info.ndim != 1) {
                     throw std::runtime_error("Expected a 1d byte buffer");
                 }
                 if (indices_info.ndim != 1) {
                     throw std::runtime_error("Expected a 1d indices buffer");
                 }
                 return new StringList((char *)bytes_info.ptr, bytes_info.shape[0], (typename StringList::index_type *)indices_info.ptr, string_count, offset);
             }),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>() // keep a reference to the ndarrays
             )
        // same ctor, duplicate code, cannot make null_bitmap accept None
        .def(py::init([](py::buffer bytes, py::array_t<typename StringList::index_type, py::array::c_style> &indices, size_t string_count, size_t offset,
                         py::array_t<uint8_t, py::array::c_style> null_bitmap, size_t null_offset) {
                 py::buffer_info bytes_info = bytes.request();
                 py::buffer_info indices_info = indices.request();
                 if (bytes_info.ndim != 1) {
                     throw std::runtime_error("Expected a 1d byte buffer");
                 }
                 if (indices_info.ndim != 1) {
                     throw std::runtime_error("Expected a 1d indices buffer");
                 }
                 uint8_t *null_bitmap_ptr = 0;
                 if (null_bitmap) {
                     py::buffer_info null_bitmap_info = null_bitmap.request();
                     if (null_bitmap_info.ndim != 1) {
                         throw std::runtime_error("Expected a 1d indices buffer");
                     }
                     null_bitmap_ptr = (uint8_t *)null_bitmap_info.ptr;
                 }
                 return new StringList((char *)bytes_info.ptr, bytes_info.shape[0], (typename StringList::index_type *)indices_info.ptr, string_count, offset, null_bitmap_ptr, null_offset);
             }),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 6>() // keep a reference to the ndarrays
             )
        .def("split", &StringList::split, py::keep_alive<0, 1>())
        .def("slice", &StringList::slice, py::keep_alive<0, 1>())
        .def("slice", &StringList::slice_byte_offset, py::keep_alive<0, 1>())
        .def("fill_from", &StringList::fill_from)
        // .def("get", (const std::string (StringList::*)(size_t))&StringList::get)
        // bug? we have to add this again
        // .def("get", (py::object (StringSequenceBase::*)(size_t, size_t))&StringSequenceBase::get, py::return_value_policy::take_ownership)
        .def_property_readonly("bytes",
                               [](const StringList &sl) {
                                   return py::array_t<char>(sl.byte_length, sl.bytes, py::cast(sl));
                               })
        .def_property_readonly("indices", [](const StringList &sl) { return py::array_t<typename StringList::index_type>(sl.length + 1, sl.indices, py::cast(sl)); })
        .def_property_readonly("null_bitmap",
                               [](const StringList &sl) -> py::object {
                                   if (sl.null_bitmap) { // TODO: what if there is a lazy view
                                       size_t length = (sl.length + 7) / 8;
                                       return py::array_t<unsigned char>(length, sl.null_bitmap);
                                   } else {
                                       return py::cast<py::none>(Py_None);
                                   }
                               })
        .def_property_readonly("offset", [](const StringList &sl) { return sl.offset; })
        .def_property_readonly("null_offset", [](const StringList &sl) { return sl.null_offset; })
        .def_property_readonly("length", [](const StringList &sl) { return sl.length; })
        // .def("__repr__",
        //     [](const StringList &sl) {
        //         return "<vaex.strings.StringList buffer='" + sl.get() + "'>";
        //     }
        // )
        ;
}

template <class T>
StringList64 *to_string(py::array_t<T, py::array::c_style> values_) {
    size_t length = values_.size();
    auto values = values_.template unchecked<1>();
    if (values_.ndim() != 1) {
        throw std::runtime_error("Expected a 1d array");
    }
    {
        py::gil_scoped_release release;
        StringList64 *sl = new StringList64(length * 2, length);
        size_t byte_offset = 0;
        for (size_t i = 0; i < length; i++) {
            std::string str = std::to_string(values(i));
            while (byte_offset + str.length() > sl->byte_length) {
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

template <class T>
StringList64 *to_string_mask(py::array_t<T, py::array::c_style> values_, py::array_t<bool, py::array::c_style> mask_) {
    size_t length = values_.size();
    auto values = values_.template unchecked<1>();
    if (values_.ndim() != 1) {
        throw std::runtime_error("Expected a 1d array");
    }

    py::buffer_info info_mask = mask_.request();
    if (info_mask.ndim != 1) {
        throw std::runtime_error("Expected a 1d byte buffer");
    }
    bool *mask = (bool *)info_mask.ptr;
    if (info_mask.size != length) {
        throw std::runtime_error("Indices and mask are of unequal length");
    }

    {
        py::gil_scoped_release release;
        StringList64 *sl = new StringList64(length * 2, length);
        size_t byte_offset = 0;
        for (size_t i = 0; i < length; i++) {
            if (mask[i] == 1) {
                if (sl->null_bitmap == nullptr)
                    sl->add_null_bitmap();
                sl->set_null(i);
                sl->indices[i] = byte_offset;
            } else {
                std::string str = std::to_string(values(i));
                while (byte_offset + str.length() > sl->byte_length) {
                    sl->grow();
                }
                std::copy(str.begin(), str.end(), sl->bytes + byte_offset);
                sl->indices[i] = byte_offset;
                byte_offset += str.length();
            }
        }
        sl->indices[length] = byte_offset;
        return sl;
    }
}

template <class T>
StringList64 *format(py::array_t<T, py::array::c_style> values_, const char *format) {
    size_t length = values_.size();
    auto values = values_.template unchecked<1>();
    if (values_.ndim() != 1) {
        throw std::runtime_error("Expected a 1d array");
    }
    {
        py::gil_scoped_release release;
        StringList64 *sl = new StringList64(length * 2, length);
        size_t byte_offset = 0;
        for (size_t i = 0; i < length; i++) {
            sl->indices[i] = byte_offset;
            bool done = false;
            int ret;
            while (!done) {
                int64_t bytes_left = sl->byte_length - byte_offset;
                ret = snprintf(sl->bytes + byte_offset, bytes_left, format, (T)values(i));
                if (ret < 0) {
                    throw std::runtime_error("Invalid format");
                } else if (ret < bytes_left) {
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

StringList64 *format_string(StringSequence *values, const char *format) {
    size_t length = values->length;
    {
        py::gil_scoped_release release;
        StringList64 *sl = new StringList64(length * 2, length);
        size_t byte_offset = 0;
        for (size_t i = 0; i < length; i++) {
            sl->indices[i] = byte_offset;
            bool done = false;
            int ret;
            while (!done) {
                int64_t bytes_left = sl->byte_length - byte_offset;
                auto value = values->get(i);
                ret = snprintf(sl->bytes + byte_offset, bytes_left, format, value.c_str());
                if (ret < 0) {
                    throw std::runtime_error("Invalid format");
                } else if (ret < bytes_left) {
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
    py::class_<StringSequence, std::shared_ptr<StringSequence>> string_sequence(m, "StringSequence");
    py::class_<StringSequenceBase, std::shared_ptr<StringSequenceBase>> string_sequence_base(m, "StringSequenceBase", string_sequence);
    string_sequence_base.def("to_numpy", &StringSequenceBase::to_numpy, py::return_value_policy::take_ownership)
        .def("lazy_index", &StringSequenceBase::lazy_index<int32_t>, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("lazy_index", &StringSequenceBase::lazy_index<int64_t>, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("lazy_index", &StringSequenceBase::lazy_index<uint32_t>, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("lazy_index", &StringSequenceBase::lazy_index<uint64_t>, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("index", &StringSequenceBase::index<bool>)
        .def("index", &StringSequenceBase::index<int32_t>)
        .def("index", &StringSequenceBase::index<int64_t>)
        .def("index", &StringSequenceBase::index<uint32_t>)
        .def("index", &StringSequenceBase::index<uint64_t>)
        // no need for a index_masked with bools, since we can logically and the two masks
        .def("index", &StringSequenceBase::index_masked<int32_t>)
        .def("index", &StringSequenceBase::index_masked<int64_t>)
        .def("index", &StringSequenceBase::index_masked<uint32_t>)
        .def("index", &StringSequenceBase::index_masked<uint64_t>)
        .def("tolist", &StringSequenceBase::tolist)
        .def("capitalize", &StringSequenceBase::capitalize, py::keep_alive<0, 1>())
        .def("concat", &StringSequenceBase::concat)
        .def("concat_reverse", &StringSequenceBase::concat_reverse)
        .def("concat", &StringSequenceBase::concat2)
        .def("pad", &StringSequenceBase::pad)
        .def("search", &StringSequenceBase::search, "Tests if strings contains pattern", py::arg("pattern"), py::arg("regex")) //, py::call_guard<py::gil_scoped_release>())
        .def("count", &StringSequenceBase::count, "Count occurrences of pattern", py::arg("pattern"), py::arg("regex"))
        .def("endswith", &StringSequenceBase::endswith)
        .def("find", &StringSequenceBase::find)
        .def("isin", &StringSequenceBase::isin)
        .def("match", &StringSequenceBase::match, "Tests if strings matches regex", py::arg("pattern"))
        .def("equals", &StringSequenceBase::equals, "Tests if strings are equal")
        .def("equals", &StringSequenceBase::equals2, "Tests if strings are equal")
        .def("lstrip", &StringSequenceBase::lstrip)
        .def("rstrip", &StringSequenceBase::rstrip)
        .def("repeat", &StringSequenceBase::repeat)
        .def("replace", &StringSequenceBase_replace)
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
            if (sl.null_bitmap) { // TODO: what if there is a lazy view
                auto ar = py::array_t<bool>(sl.length);
                auto ar_unsafe = ar.mutable_unchecked<1>();
                {
                    py::gil_scoped_release release;
                    for (size_t i = 0; i < sl.length; i++) {
                        ar_unsafe(i) = sl.is_null(i);
                    }
                }
                return std::move(ar);
            } else {
                return py::cast<py::none>(Py_None);
            }
        });
    py::class_<StringListList>(m, "StringListList")
        .def("all", &StringListList::all)
        .def("get", &StringListList::get)
        .def("join", &StringListList::join)
        .def("get", &StringListList::getlist)
        .def("print", &StringListList::print)
        .def("__len__", [](const StringListList &obj) { return obj.length; });
    add_string_list<StringList32>(m, string_sequence_base, "StringList32");
    add_string_list<StringList64>(m, string_sequence_base, "StringList64");
    py::class_<StringArray, std::shared_ptr<StringArray>>(m, "StringArray", string_sequence_base)
        .def(py::init([](py::buffer string_array) {
            py::buffer_info info = string_array.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Expected a 1d byte buffer");
            }
            if (info.format != "O") {
                throw std::runtime_error("Expected an object array");
            }
            // std::cout << info.format << " format" << std::endl;
            return std::unique_ptr<StringArray>(new StringArray((PyObject **)info.ptr, info.shape[0]));
        }) // no need to keep a reference to the ndarrays
             )
        .def(py::init([](py::buffer string_array, py::buffer mask_array) {
            py::buffer_info info = string_array.request();
            py::buffer_info mask_info = mask_array.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Expected a 1d byte buffer");
            }
            if (info.format != "O") {
                throw std::runtime_error("Expected an object array");
            }
            // std::cout << info.format << " format" << std::endl;
            return std::unique_ptr<StringArray>(new StringArray((PyObject **)info.ptr, info.shape[0], (uint8_t *)mask_info.ptr));
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
                for (size_t i = 0; i < sl.length; i++) {
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
        });
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

    // with mask
    m.def("to_string", &to_string_mask<float>);
    m.def("to_string", &to_string_mask<double>);
    m.def("to_string", &to_string_mask<int64_t>);
    m.def("to_string", &to_string_mask<int32_t>);
    m.def("to_string", &to_string_mask<int16_t>);
    m.def("to_string", &to_string_mask<int8_t>);
    m.def("to_string", &to_string_mask<uint64_t>);
    m.def("to_string", &to_string_mask<uint32_t>);
    m.def("to_string", &to_string_mask<uint16_t>);
    m.def("to_string", &to_string_mask<uint8_t>);
    m.def("to_string", &to_string_mask<bool>);

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
    m.def("join", &join<StringList32>);
    m.def("join", &join<StringList64>);
}

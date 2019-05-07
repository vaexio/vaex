#include <nonstd/string_view.hpp>
#include <string>
typedef nonstd::string_view string_view;
typedef std::string string;

inline bool _is_null(uint8_t* null_bitmap, size_t i) {
    if(null_bitmap) {
        size_t byte_index = i / 8;
        size_t bit_index = (i % 8);
        return (null_bitmap[byte_index] & (1 << bit_index)) == 0;
    } else {
        return false;
    }
}


class StringSequence {
    public:
    StringSequence(size_t length, uint8_t* null_bitmap=nullptr, int64_t null_offset=0) : length(length), null_bitmap(null_bitmap), null_offset(null_offset) {
    }
    virtual ~StringSequence() {
    }
    virtual string_view view(int64_t i) const = 0;
    virtual const std::string get(int64_t i) const = 0;
    virtual size_t byte_size() const = 0;
    virtual bool is_null(int64_t i) const {
        return _is_null(null_bitmap, i + null_offset);
    }
    virtual bool has_null() const {
        return null_bitmap != nullptr;
    }
    size_t length;
    uint8_t* null_bitmap;
    int64_t null_offset;
};
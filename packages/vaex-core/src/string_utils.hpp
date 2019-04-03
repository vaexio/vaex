/* regex match/search using string_view */
#include <nonstd/string_view.hpp>

typedef nonstd::string_view string_view;

/*
class string_view {
public:
    string_view(const char* ptr, size_t count) : ptr(ptr), count(count) {  }
    string_view(const char* ptr) : ptr(ptr), count(0) { }
    typedef const char* const_iterator;
    const char* begin() { return ptr; }
    const char* end() { return ptr + count; }
    size_t length() const { return count; }
    size_t find(const std::string& pattern) {
        return std::string(begin(), end()).find(pattern);
    }
    const char* ptr;
    size_t count;
};
*/

using svmatch = std::match_results<string_view::const_iterator>;
using svsub_match = std::sub_match<string_view::const_iterator>;

inline string_view get_sv(const svsub_match& m) {
    return string_view(m.first, m.length());
}

inline bool regex_match(string_view sv,
                  svmatch& m,
                  const std::regex& e,
                  std::regex_constants::match_flag_type flags =
                      std::regex_constants::match_default) {
    return std::regex_match(sv.begin(), sv.end(), m, e, flags);
}

inline bool regex_match(string_view sv,
                  const std::regex& e,
                  std::regex_constants::match_flag_type flags =
                      std::regex_constants::match_default) {
    return std::regex_match(sv.begin(), sv.end(), e, flags);
}
inline bool regex_search(string_view sv,
                  const std::regex& e,
                  std::regex_constants::match_flag_type flags =
                      std::regex_constants::match_default) {
    return std::regex_search(sv.begin(), sv.end(), e, flags);
}
#ifdef VAEX_REGEX_USE_BOOST
inline bool regex_search(string_view sv,
                  const boost::regex& e,
                  boost::regex_constants::match_flag_type flags =
                      boost::regex_constants::match_default) {
    return boost::regex_search(sv.begin(), sv.end(), e, flags);
}
#endif

struct stripper {
    std::string chars;
    bool left, right;
    stripper(std::string chars, bool left, bool right) : chars(chars), left(left), right(right) {}
    void operator()(const string_view& source, char*& target) {
        size_t length = source.length();
        auto begin = source.begin();
        auto end = source.end();
        if(left && length > 0) {
            if(chars.length()) {
                while(chars.find(*begin) != std::string::npos && length > 0) {
                    begin++;
                    length--;
                }
            } else {
                while(::isspace(*begin) && length > 0) {
                    begin++;
                    length--;
                }
            }
        }
        if(right && length > 0) {
            end--;
            if(chars.length()) {
                while(chars.find(*end) != std::string::npos && length > 0) {
                    end--;
                    length--;
                }
            } else {
                while(::isspace(*end) && length > 0) {
                    end--;
                    length--;
                }
            }
            end++;
        }
        if(length) {
            std::copy(begin, end, target);
            target += length;
        }

    }
};


inline int64_t str_len(const string_view& source) {
    const char *str = source.begin();
    const char *end = source.end();
    int64_t string_length = 0;
    while(str < end) {
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
        string_length += 1;
    }
    return string_length;
}

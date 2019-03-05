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
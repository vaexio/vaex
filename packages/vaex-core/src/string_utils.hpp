/* regex match/search using string_view */
#include <string_view>

using svmatch = std::match_results<std::string_view::const_iterator>;
using svsub_match = std::sub_match<std::string_view::const_iterator>;

inline std::string_view get_sv(const svsub_match& m) {
    return std::string_view(m.first, m.length());
}

inline bool regex_match(std::string_view sv,
                  svmatch& m,
                  const std::regex& e,
                  std::regex_constants::match_flag_type flags = 
                      std::regex_constants::match_default) {
    return std::regex_match(sv.begin(), sv.end(), m, e, flags);
}

inline bool regex_match(std::string_view sv,
                  const std::regex& e,
                  std::regex_constants::match_flag_type flags = 
                      std::regex_constants::match_default) {
    return std::regex_match(sv.begin(), sv.end(), e, flags);
}

inline bool regex_search(std::string_view sv,
                  const std::regex& e,
                  std::regex_constants::match_flag_type flags = 
                      std::regex_constants::match_default) {
    return std::regex_search(sv.begin(), sv.end(), e, flags);
}
/* regex match/search using string_view */
#include <nonstd/string_view.hpp>
#include <regex>

const char REPLACEMENT_CHAR = '?';
const char32_t CHARS = 0x110000;
extern const char32_t othercase_block[][256];
extern const uint8_t category_index[CHARS >> 8];
extern const uint8_t category_block[][256];
const extern uint8_t othercase_index[CHARS >> 8];
enum othercase_type { LOWER_ONLY = 1, UPPERTITLE_ONLY = 2, LOWER_THEN_UPPER = 3, UPPER_THEN_TITLE = 4, TITLE_THEN_LOWER = 5 };

typedef uint32_t category_t;
enum : uint8_t {
    _Lu = 1,
    _Ll = 2,
    _Lt = 3,
    _Lm = 4,
    _Lo = 5,
    _Mn = 6,
    _Mc = 7,
    _Me = 8,
    _Nd = 9,
    _Nl = 10,
    _No = 11,
    _Pc = 12,
    _Pd = 13,
    _Ps = 14,
    _Pe = 15,
    _Pi = 16,
    _Pf = 17,
    _Po = 18,
    _Sm = 19,
    _Sc = 20,
    _Sk = 21,
    _So = 22,
    _Zs = 23,
    _Zl = 24,
    _Zp = 25,
    _Cc = 26,
    _Cf = 27,
    _Cs = 28,
    _Co = 29,
    _Cn = 30
};
enum : category_t {
    Lu = 1 << _Lu,
    Ll = 1 << _Ll,
    Lt = 1 << _Lt,
    Lut = Lu | Lt,
    LC = Lu | Ll | Lt,
    Lm = 1 << _Lm,
    Lo = 1 << _Lo,
    L = Lu | Ll | Lt | Lm | Lo,
    Mn = 1 << _Mn,
    Mc = 1 << _Mc,
    Me = 1 << _Me,
    M = Mn | Mc | Me,
    Nd = 1 << _Nd,
    Nl = 1 << _Nl,
    No = 1 << _No,
    N = Nd | Nl | No,
    Pc = 1 << _Pc,
    Pd = 1 << _Pd,
    Ps = 1 << _Ps,
    Pe = 1 << _Pe,
    Pi = 1 << _Pi,
    Pf = 1 << _Pf,
    Po = 1 << _Po,
    P = Pc | Pd | Ps | Pe | Pi | Pf | Po,
    Sm = 1 << _Sm,
    Sc = 1 << _Sc,
    Sk = 1 << _Sk,
    So = 1 << _So,
    S = Sm | Sc | Sk | So,
    Zs = 1 << _Zs,
    Zl = 1 << _Zl,
    Zp = 1 << _Zp,
    Z = Zs | Zl | Zp,
    Cc = 1 << _Cc,
    Cf = 1 << _Cf,
    Cs = 1 << _Cs,
    Co = 1 << _Co,
    Cn = 1 << _Cn,
    C = Cc | Cf | Cs | Co | Cn
};

static const int32_t DEFAULT_CAT = Cn;

inline category_t char32_category(char32_t chr) { return chr < CHARS ? 1 << category_block[category_index[chr >> 8]][chr & 0xFF] : DEFAULT_CAT; }
inline bool char32_isalpha(char32_t chr) {
    category_t cat = char32_category(chr);
    return (cat == Lm) || (cat == Lt) || (cat == Lu) || (cat == Ll) || (cat == Lo);
}

inline bool char32_isalnum(char32_t chr) {
    category_t cat = char32_category(chr);
    bool isalpha = (cat == Lm) || (cat == Lt) || (cat == Lu) || (cat == Ll) || (cat == Lo);
    // not sure how Python distinguises the 3 cases
    // bool isdecimal = (cat == Nd);
    // bool isdigit = (cat == Nd) || (cat == Nl) || (cat == No);
    // bool isnumeric =
    // but this is more clear
    bool isnum = (cat == Nd) || (cat == Nl) || (cat == No);
    return isalpha || isnum;
}
inline char32_t char32_uppercase(char32_t chr) {
    if (chr < CHARS) {
        char32_t othercase = othercase_block[othercase_index[chr >> 8]][chr & 0xFF];
        if ((othercase & 0xFF) == othercase_type::UPPERTITLE_ONLY)
            return othercase >> 8;
        if ((othercase & 0xFF) == othercase_type::UPPER_THEN_TITLE)
            return othercase >> 8;
        if ((othercase & 0xFF) == othercase_type::LOWER_THEN_UPPER)
            return othercase_block[othercase_index[(othercase >> 8) >> 8]][(othercase >> 8) & 0xFF] >> 8;
    }
    return chr;
}

inline char32_t char32_lowercase(char32_t chr) {
    if (chr < CHARS) {
        char32_t othercase = othercase_block[othercase_index[chr >> 8]][chr & 0xFF];
        if ((othercase & 0xFF) == othercase_type::LOWER_ONLY)
            return othercase >> 8;
        if ((othercase & 0xFF) == othercase_type::LOWER_THEN_UPPER)
            return othercase >> 8;
        if ((othercase & 0xFF) == othercase_type::TITLE_THEN_LOWER)
            return othercase_block[othercase_index[(othercase >> 8) >> 8]][(othercase >> 8) & 0xFF] >> 8;
    }
    return chr;
}

inline bool utf8_isupper(char32_t chr) { return chr == char32_uppercase(chr); }

inline bool utf8_islower(char32_t chr) { return chr == char32_lowercase(chr); }

inline char32_t utf8_decode(const char *&str, size_t &len) {
    if (!len)
        return 0;
    --len;
    if (((unsigned char)*str) < 0x80)
        return (unsigned char)*str++;
    else if (((unsigned char)*str) < 0xC0)
        return ++str, REPLACEMENT_CHAR;
    else if (((unsigned char)*str) < 0xE0) {
        char32_t res = (((unsigned char)*str++) & 0x1F) << 6;
        if (len <= 0 || ((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        return res + ((--len, ((unsigned char)*str++)) & 0x3F);
    } else if (((unsigned char)*str) < 0xF0) {
        char32_t res = (((unsigned char)*str++) & 0x0F) << 12;
        if (len <= 0 || ((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        res += ((--len, ((unsigned char)*str++)) & 0x3F) << 6;
        if (len <= 0 || ((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        return res + ((--len, ((unsigned char)*str++)) & 0x3F);
    } else if (((unsigned char)*str) < 0xF8) {
        char32_t res = (((unsigned char)*str++) & 0x07) << 18;
        if (len <= 0 || ((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        res += ((--len, ((unsigned char)*str++)) & 0x3F) << 12;
        if (len <= 0 || ((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        res += ((--len, ((unsigned char)*str++)) & 0x3F) << 6;
        if (len <= 0 || ((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        return res + ((--len, ((unsigned char)*str++)) & 0x3F);
    } else
        return ++str, REPLACEMENT_CHAR;
}

inline char32_t utf8_decode(const char *&str) {
    if (((unsigned char)*str) < 0x80)
        return (unsigned char)*str++;
    else if (((unsigned char)*str) < 0xC0)
        return ++str, REPLACEMENT_CHAR;
    else if (((unsigned char)*str) < 0xE0) {
        char32_t res = (((unsigned char)*str++) & 0x1F) << 6;
        if (((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        return res + (((unsigned char)*str++) & 0x3F);
    } else if (((unsigned char)*str) < 0xF0) {
        char32_t res = (((unsigned char)*str++) & 0x0F) << 12;
        if (((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        res += (((unsigned char)*str++) & 0x3F) << 6;
        if (((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        return res + (((unsigned char)*str++) & 0x3F);
    } else if (((unsigned char)*str) < 0xF8) {
        char32_t res = (((unsigned char)*str++) & 0x07) << 18;
        if (((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        res += (((unsigned char)*str++) & 0x3F) << 12;
        if (((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        res += (((unsigned char)*str++) & 0x3F) << 6;
        if (((unsigned char)*str) < 0x80 || ((unsigned char)*str) >= 0xC0)
            return REPLACEMENT_CHAR;
        return res + (((unsigned char)*str++) & 0x3F);
    } else
        return ++str, REPLACEMENT_CHAR;
}

inline void utf8_append(char *&str, char32_t chr) {
    if (chr < 0x80)
        *str++ = chr;
    else if (chr < 0x800) {
        *str++ = 0xC0 + (chr >> 6);
        *str++ = 0x80 + (chr & 0x3F);
    } else if (chr < 0x10000) {
        *str++ = 0xE0 + (chr >> 12);
        *str++ = 0x80 + ((chr >> 6) & 0x3F);
        *str++ = 0x80 + (chr & 0x3F);
    } else if (chr < 0x200000) {
        *str++ = 0xF0 + (chr >> 18);
        *str++ = 0x80 + ((chr >> 12) & 0x3F);
        *str++ = 0x80 + ((chr >> 6) & 0x3F);
        *str++ = 0x80 + (chr & 0x3F);
    } else
        *str++ = REPLACEMENT_CHAR;
}

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

inline string_view get_sv(const svsub_match &m) { return string_view(m.first, m.length()); }

inline bool regex_match(string_view sv, svmatch &m, const std::regex &e, std::regex_constants::match_flag_type flags = std::regex_constants::match_default) {
    return std::regex_match(sv.begin(), sv.end(), m, e, flags);
}

inline bool regex_match(string_view sv, const std::regex &e, std::regex_constants::match_flag_type flags = std::regex_constants::match_default) {
    return std::regex_match(sv.begin(), sv.end(), e, flags);
}
inline bool regex_search(string_view sv, const std::regex &e, std::regex_constants::match_flag_type flags = std::regex_constants::match_default) {
    return std::regex_search(sv.begin(), sv.end(), e, flags);
}
#ifdef VAEX_REGEX_USE_BOOST
inline bool regex_search(string_view sv, const boost::regex &e, boost::regex_constants::match_flag_type flags = boost::regex_constants::match_default) {
    return boost::regex_search(sv.begin(), sv.end(), e, flags);
}
#endif

struct stripper {
    std::string chars;
    bool left, right;
    stripper(std::string chars, bool left, bool right) : chars(chars), left(left), right(right) {}
    void operator()(const string_view &source, char *&target) {
        size_t length = source.length();
        auto begin = source.begin();
        auto end = source.end();
        if (left && length > 0) {
            if (chars.length()) {
                while (chars.find(*begin) != std::string::npos && length > 0) {
                    begin++;
                    length--;
                }
            } else {
                while (::isspace(*begin) && length > 0) {
                    begin++;
                    length--;
                }
            }
        }
        if (right && length > 0) {
            end--;
            if (chars.length()) {
                while (chars.find(*end) != std::string::npos && length > 0) {
                    end--;
                    length--;
                }
            } else {
                while (::isspace(*end) && length > 0) {
                    end--;
                    length--;
                }
            }
            end++;
        }
        if (length) {
            std::copy(begin, end, target);
            target += length;
        }
    }
};

inline int64_t str_len(const string_view &source) {
    const char *str = source.begin();
    const char *end = source.end();
    int64_t string_length = 0;
    while (str < end) {
        char current = *str;
        if (((unsigned char)current) < 0x80) {
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

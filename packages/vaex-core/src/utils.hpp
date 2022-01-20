#ifndef VAEX_UTILS_H
#define VAEX_UTILS_H

namespace vaex {
template <typename T>
struct FixBool {
    using value = T;
};

template <>
struct FixBool<bool> {
    using value = uint8_t;
};

template <template <typename...> class Generator, typename... Args>
void all_types(Args... args) {
    Generator<double>("float64")();
    // Generator<double>("float64", ... args);
    // Generator<float>("float32", ... args);
    // Generator<int64_t>("int64", ... args);
    // Generator<int32_t>("int32", ... args);
    // Generator<int16_t>("int16", ... args);
    // Generator<int8_t>("int8", ... args);
    // Generator<uint64_t>("uint64", ... args);
    // Generator<uint32_t>("uint32", ... args);
    // Generator<uint16_t>("uint16", ... args);
    // Generator<uint8_t>("uint8", ... args);
    // Generator<bool>("bool", ... args);
}

template <typename T>
struct type_name {
    constexpr static const char *value = "unknown";
};

template <>
struct type_name<double> {
    constexpr static const char *value = "float64";
};

template <>
struct type_name<float> {
    constexpr static const char *value = "float32";
};

template <>
struct type_name<int64_t> {
    constexpr static const char *value = "int64";
};

template <>
struct type_name<int32_t> {
    constexpr static const char *value = "int32";
};

template <>
struct type_name<int16_t> {
    constexpr static const char *value = "int16";
};

template <>
struct type_name<int8_t> {
    constexpr static const char *value = "int8";
};

template <>
struct type_name<uint64_t> {
    constexpr static const char *value = "uint64";
};

template <>
struct type_name<uint32_t> {
    constexpr static const char *value = "uint32";
};

template <>
struct type_name<uint16_t> {
    constexpr static const char *value = "uint16";
};

template <>
struct type_name<uint8_t> {
    constexpr static const char *value = "uint8";
};

template <>
struct type_name<bool> {
    constexpr static const char *value = "bool";
};

} // namespace vaex
#endif // VAEX_UTILS_H

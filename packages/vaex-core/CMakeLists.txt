# Use this like:
#  (cd packages/vaex-core; VAEX_BUILD_SKBUILD=1 python setup.py build_ext --build-type=release --inplace --verbose -- -- -j 20)
# currently experimental, and used for development only, since it makes rebuilding much faster

cmake_minimum_required(VERSION 3.11.0)
project(hello)
find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)



IF(CMAKE_COMPILER_IS_GNUCC)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fdiagnostics-color -fmax-errors=4 -std=c++11 -Werror=return-type -Wno-unused-parameter -O3 -funroll-loops")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

if(MSVC)
    add_compile_options("/bigobj")
endif()

# this doesn't seem to work:
#find_package(pybind11 REQUIRED)
#message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}/pybind11")
# so we do it this way:
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(WARNING "Found numpy: ${NumPy_INCLUDE_DIRS}")

# string module
add_library("superstrings" MODULE
    "src/strings.cpp"
    "src/string_utils.cpp"
)
target_include_directories("superstrings" PUBLIC
    ${NumPy_INCLUDE_DIRS}
    "vendor/string-view-lite/include"
    "vendor/pybind11/include"
    "vendor/boost"
    "${PYTHON_PREFIX}/include"
)


target_link_directories("superstrings" PUBLIC
    "${PYTHON_PREFIX}/lib")
target_link_libraries("superstrings" pcre pcrecpp)

# aggregation library
add_library("superagg" MODULE
    "src/agg_nunique_string.cpp"
    "src/agg_minmax.cpp"
    "src/agg_nunique.cpp"
    "src/agg_sum.cpp"
    "src/agg_first.cpp"
    "src/agg_list.cpp"
    "src/agg_count.cpp"
    "src/agg.cpp"
    "src/binner_combined.cpp"
    "src/binner_ordinal.cpp"
    "src/binner_hash.cpp"
    "src/binners.cpp"
    "src/string_utils.cpp"
)
target_include_directories("superagg" PUBLIC
    ${NumPy_INCLUDE_DIRS}
    "vendor/string-view-lite/include"
    "vendor/pybind11/include"
    "vendor/hopscotch-map/include"
)

# util/hash library
add_library("superutils" MODULE
    "src/hash_string.cpp"
    "src/hash_primitives_pot.cpp"
    "src/hash_object.cpp"
    "src/hash_primitives_prime.cpp"
    "src/superutils.cpp"
    "src/string_utils.cpp"
    "src/string_utils.cpp"
)
target_include_directories("superutils" PUBLIC
    ${NumPy_INCLUDE_DIRS}
    "vendor/abseil-cpp"
    "vendor/flat_hash_map"
    "vendor/sparse-map/include"
    "vendor/hopscotch-map/include"
    "vendor/string-view-lite/include"
    "vendor/pybind11/include"
)
target_link_directories("superutils" PUBLIC
    "${PYTHON_PREFIX}/lib")
target_link_libraries("superutils" pcre pcrecpp)


add_compile_definitions(VAEX_USE_TSL)


python_extension_module("superagg")
python_extension_module("superstrings")
python_extension_module("superutils")
install(TARGETS "superagg" "superstrings" "superutils" LIBRARY DESTINATION vaex)
# install(TARGETS "superagg" LIBRARY DESTINATION vaex)

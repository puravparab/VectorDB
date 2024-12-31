cmake_minimum_required(VERSION 3.16)
project(VectorDB LANGUAGES CXX)

add_library(
    vectordb_cpp
    src/vector/vector.cpp
    src/index/brute_force.cpp
    src/index/hnsw.cpp
)
target_include_directories(vectordb_cpp PUBLIC src)
target_compile_features(vectordb_cpp PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(vectordb_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_executable(vectordb_cpp_tests tests/cpp/test_indexes.cpp)
target_link_libraries(vectordb_cpp_tests PRIVATE vectordb_cpp)
add_test(NAME vectordb_cpp_tests COMMAND vectordb_cpp_tests)

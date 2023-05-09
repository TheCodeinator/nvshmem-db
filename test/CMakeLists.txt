project(Tests LANGUAGES CXX)

# TODO: use unit testing framework such as Boost Test, Google Test or Catch2

add_executable(TestA src/testA.cpp)
target_link_libraries(TestA PRIVATE nvshmem-db)
include_directories(../src)
add_test(NAME TestA COMMAND TestA)

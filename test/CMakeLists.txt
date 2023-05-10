project(Tests LANGUAGES CXX)

# TODO: use unit testing framework such as Boost Test, Google Test or Catch2

# include the header files from our library
include_directories(${CMAKE_SOURCE_DIR}/inc)
# add cpp files to the test cases
add_executable(TestA src/testA.cpp)
# link our own library
target_link_libraries(TestA PRIVATE nvshmem-db)
# add a test with a name and specify the target to run
add_test(NAME TestA COMMAND TestA)

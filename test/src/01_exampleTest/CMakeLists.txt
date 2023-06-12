#project(Tests LANGUAGES CXX)

# TODO: use unit testing framework such as Boost Test, Google Test or Catch2

# include the header files from our library
include_directories(${CMAKE_SOURCE_DIR}/inc)
# add cpp files to the test cases
add_executable(01_exampleTest exampleTest.cpp)
# link our own library
target_link_libraries(01_exampleTest PRIVATE nvshmem-db)
# add a test with a name and specify the target to run
add_test(NAME 01_exampleTest COMMAND 01_exampleTest)

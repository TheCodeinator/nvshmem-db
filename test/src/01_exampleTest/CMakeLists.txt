#project(Tests LANGUAGES CXX)

# TODO: use unit testing framework such as Boost Test, Google Test or Catch2

# add cpp files to the test cases
add_executable(01_exampleTest exampleTest.cpp)
add_dependencies(01_exampleTest nvshmem-db)
# link our own library
target_link_libraries(01_exampleTest nvshmem-db)
# add a test with a name and specify the target to run
add_test(NAME 01_exampleTest COMMAND 01_exampleTest)

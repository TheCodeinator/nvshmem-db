add_executable(04_baselineTest baselineTest.cpp)
add_dependencies(04_baselineTest baseline-db)

target_link_libraries(04_baselineTest baseline-db)

set_property(TARGET 04_baselineTest PROPERTY POSITION_INDEPENDENT_CODE ON)

SET(TEST_COMMAND
        /opt/hydra/bin/nvshmrun -np 2
        $<TARGET_FILE:04_baselineTest>)

add_test(NAME 04_baselineTest COMMAND ${TEST_COMMAND})

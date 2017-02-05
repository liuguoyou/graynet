set(SRCS
	TestNodes.cpp
)

add_source_groups(${SRCS})

add_executable(graynet_test ${SRCS})

target_link_libraries(graynet_test graynet gtest gtest_main)

set(SRCS
	MNIST.cpp
)

add_source_groups(${SRCS})

add_executable(mnist ${SRCS})

target_link_libraries(mnist graynet)


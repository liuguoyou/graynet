set(SRCS
	Device.h
	Device.cpp
	Expression.h
	Expression.cpp
	Expression_p.h
	Graph.h
	Graph.cpp
	Layer.h
	Layer.cpp
	Node.h
	Node.cpp
	Optimizer.h
	Optimizer.cpp
	Shape.h
	Shape.cpp
	Tensor.h
	Tensor.cpp
	Utils.h
	Utils.cpp
	graynet.h
)

add_source_groups(${SRCS})

if(${CUDA_FOUND})
	set(SRCS ${SRCS};Expression.cu)
	add_definitions(-DUSE_CUDA)
	include_directories(${CUDNN_ROOT}/include)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture=compute_35 --gpu-code=sm_35,sm_50,sm_52")
	cuda_add_library(graynet STATIC ${SRCS})
	if(MSVC)
		set(CUDNN_LIBRARIES ${CUDNN_ROOT}/lib/x64/cudnn.lib)
	else()
		set(CUDNN_LIBRARIES ${CUDNN_ROOT}/lib64/libcudnn.so)
	endif()
else()
	add_library(graynet STATIC ${SRCS})
endif()

cuda_add_cublas_to_target(graynet)

target_link_libraries(graynet ${BLAS_LIBRARIES} ${CUDNN_LIBRARIES})

target_include_directories(graynet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

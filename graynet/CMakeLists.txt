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
	initializer_view.h
	graynet.h
)

add_source_groups(${SRCS})

if(${GRAYNET_USE_CUDA})
	set(SRCS ${SRCS};Expression.cu)
	set_source_files_properties(Optimizer.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
	add_definitions(-DUSE_CUDA)
	include_directories(${CUDNN_ROOT}/include)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture=compute_35 --gpu-code=compute_35,sm_35 --expt-extended-lambda")
	if(NOT MSVC)
		set(CUDA_PROPAGATE_HOST_FLAGS OFF)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
	endif()
	cuda_add_library(graynet STATIC ${SRCS})
	cuda_add_cublas_to_target(graynet)
	if(MSVC)
		set(CUDNN_LIBRARIES ${CUDNN_ROOT}/lib/x64/cudnn.lib)
	else()
		set(CUDNN_LIBRARIES ${CUDNN_ROOT}/lib64/libcudnn.so)
	endif()
else()
	add_library(graynet STATIC ${SRCS})
endif()

target_link_libraries(graynet ${BLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})

target_include_directories(graynet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

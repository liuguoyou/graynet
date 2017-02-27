cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)

project(graynet)

option(GRAYNET_USE_CUDA "Use CUDA" ON)
option(GRAYNET_BUILD_EXAMPLES "Build examples" ON)
option(GRAYNET_BUILD_TESTS "Build tests" ON)

if(${GRAYNET_USE_CUDA})
	find_package(CUDA REQUIRED)
endif()

# Infer BLAS include dir
if(MSVC)
	# Set -DBLAS_LIBRARIES=`OpenBLAS dir`
	get_filename_component(BLAS_BASE_DIR ${BLAS_LIBRARIES} DIRECTORY)
	include_directories(${BLAS_BASE_DIR}/../include)
else()
	find_package(BLAS REQUIRED)
endif()

message("BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

include(cmake/AddSourceGroups.cmake)

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14")
endif()

add_subdirectory(graynet)
add_subdirectory(3rdparty)

if(${GRAYNET_BUILD_EXAMPLES})
	add_subdirectory(examples)
endif()

if(${GRAYNET_BUILD_TESTS})
	add_subdirectory(test)
endif()

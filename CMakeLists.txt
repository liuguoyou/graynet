cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)

project(graynet)

find_package(CUDA)

# Set -DBLAS_LIBRARIES=`OpenBLAS dir`
find_package(BLAS REQUIRED)

# Infer BLAS include dir
if(MSVC)
	get_filename_component(BLAS_BASE_DIR ${BLAS_LIBRARIES} DIRECTORY)
	include_directories(${BLAS_BASE_DIR}/../include)
endif()

include(cmake/AddSourceGroups.cmake)

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
endif()

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)

add_subdirectory(graynet)
add_subdirectory(3rdparty)

if(${BUILD_EXAMPLES})
	add_subdirectory(examples)
endif()

if(${BUILD_TESTS})
	add_subdirectory(test)
endif()

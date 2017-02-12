#pragma once

#ifdef _MSC_VER
#define DEBUG_BREAK()	__debugbreak()
#else
#include <signal.h>
#define DEBUG_BREAK()	raise(SIGTRAP)
#endif

#ifdef USE_CUDA
#include <cstdio>

#define CUDA_CALL(stmt)	\
	do { \
		cudaError_t cuda_status = (stmt); \
		if (cuda_status != cudaSuccess) { \
			fprintf(stderr, "%s failed, error: %s\n", #stmt, cudaGetErrorString(cuda_status)); \
			DEBUG_BREAK(); \
		} \
	} while (0)

#define CUBLAS_CALL(stmt) \
	do { \
		cublasStatus_t cublas_status = (stmt); \
		if (cublas_status != CUBLAS_STATUS_SUCCESS) { \
			fprintf(stderr, "%s failed, error: %s\n", #stmt, GetCuBLASErrorString((int)cublas_status)); \
			DEBUG_BREAK(); \
		} \
	} while (0)

const char *GetCuBLASErrorString(int status);

#define CUSPARSE_CALL(stmt) \
	do { \
		cusparseStatus_t cusparse_status = (stmt); \
		if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { \
			fprintf(stderr, "%s failed, error: %s\n", #stmt, GetCuSPARSEErrorString((int)cusparse_status)); \
			DEBUG_BREAK(); \
		} \
	} while (0)

const char *GetCuSPARSEErrorString(int status);

#define CUDNN_CALL(stmt) \
	do { \
		cudnnStatus_t cudnn_status = (stmt); \
		if (cudnn_status != CUDNN_STATUS_SUCCESS) { \
			fprintf(stderr, "%s failed, error: %s\n", #stmt, cudnnGetErrorString(cudnn_status)); \
			DEBUG_BREAK(); \
		} \
	} while (0)

#endif

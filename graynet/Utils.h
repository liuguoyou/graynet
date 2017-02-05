#pragma once

#ifdef USE_CUDA
#include <cstdio>

#define CUDA_CALL(stmt)	\
	do { \
		cudaError_t cuda_status = (stmt); \
		if (cuda_status != cudaSuccess) { \
			fprintf(stderr, "%s failed, error: %s\n", #stmt, cudaGetErrorString(cuda_status)); \
			__debugbreak(); \
		} \
	} while (0)

#define CUBLAS_CALL(stmt) \
	do { \
		cublasStatus_t cublas_status = (stmt); \
		if (cublas_status != CUBLAS_STATUS_SUCCESS) { \
			fprintf(stderr, "%s failed, error: %s\n", #stmt, GetCuBLASErrorString((int)cublas_status)); \
			__debugbreak(); \
		} \
	} while (0)

const char *GetCuBLASErrorString(int status);

#define CUDNN_CALL(stmt) \
	do { \
		cudnnStatus_t cudnn_status = (stmt); \
		if (cudnn_status != CUDNN_STATUS_SUCCESS) { \
			fprintf(stderr, "%s failed, error: %s\n", #stmt, cudnnGetErrorString(cudnn_status)); \
			__debugbreak(); \
		} \
	} while (0)

#endif

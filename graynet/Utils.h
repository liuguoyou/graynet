#pragma once

/*! This is an internal header file */

/*! \cond NOSHOW */

#include <type_traits>

#ifdef _MSC_VER
#define FORCEINLINE		__forceinline
#else
#define FORCEINLINE
#endif

namespace detail {
	template <class F, class Tuple, std::size_t... I>
	constexpr decltype(auto) apply_impl(F &&f, Tuple &&t, std::index_sequence<I...>) {
		return std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))...);
	}
}  // namespace detail

template <class F, class Tuple>
constexpr decltype(auto) apply(F &&f, Tuple &&t) {
	return detail::apply_impl(
		std::forward<F>(f), std::forward<Tuple>(t),
		std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}

#ifdef _MSC_VER
#define DEBUG_BREAK()	__debugbreak()
#else
#include <signal.h>
#define DEBUG_BREAK()	raise(SIGTRAP)
#endif

[[noreturn]] void ReportError(const char *msg, ...);

#define REPORT_ERROR(msg, ...) ReportError(msg, __VA_ARGS__)

// Simple rule on whether to use DEBUG_BREAK() or REPORT_ERROR()
// * If it means a bug in graynet, use DEBUG_BREAK().
// * If it is outside our control, use REPORT_ERROR() with a proper error message.

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

/*! \endcond */

#pragma once

#include "Node.h"

#include <cmath>
#ifdef USE_CUDA
#include <host_defines.h>
#else
#define __device__
#define __host__
#endif

template<DeviceType DeviceType, typename ForwardFunc, typename BackwardFunc>
struct BinaryOpNodeFactory {
	Node *Create(int lhs_node, int rhs_node);
};

template<DeviceType DeviceType, typename ForwardFunc, typename BackwardFunc>
struct BinaryLeftScalarOpNodeFactory {
	Node *Create(float lhs_scalar, int rhs_node);
};

template<DeviceType DeviceType, typename ForwardFunc, typename BackwardFunc>
struct BinaryRightScalarOpNodeFactory {
	Node *Create(int lhs_node, float rhs_scalar);
};

template<DeviceType DeviceType, typename ForwardFunc, typename BackwardFunc>
struct UnaryOpNodeFactory {
	Node *Create(int node);
};

template<typename Dummy, DeviceType DeviceType>
struct ReduceSumNodeFactory {
	Node *Create(int node);
};

template<typename Dummy, DeviceType DeviceType>
struct SoftmaxNodeFactory {
	Node *Create(int node);
};

template<typename Dummy, DeviceType DeviceType>
struct CrossEntropyNodeFactory {
	Node *Create(Graph *graph, int node, const std::vector<int> &labels);
};

template<typename Dummy, DeviceType DeviceType>
struct ClassificationAccuracyNodeFactory {
	Node *Create(Graph *graph, int node, const std::vector<int> &labels);
};

#define DEFINE_FUNCTOR(name, ret_type, ...) \
	struct name {	\
		inline __host__ __device__ ret_type operator()(__VA_ARGS__); \
	}; \
	inline __host__ __device__ ret_type name::operator()(__VA_ARGS__)

// Binary functors

struct BinaryNoBackward {
	inline __host__ __device__ void operator()(float lhs, float rhs, float y, float *dYdL, float *dYdR) {}
};

DEFINE_FUNCTOR(ElemAddForward, float, float lhs, float rhs) { return lhs + rhs; }
DEFINE_FUNCTOR(ElemAddBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) { *dYdL = *dYdR = 1; }

DEFINE_FUNCTOR(ElemSubForward, float, float lhs, float rhs) { return lhs - rhs; }
DEFINE_FUNCTOR(ElemSubBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) { *dYdL = 1; *dYdR = -1; }

DEFINE_FUNCTOR(ElemMulForward, float, float lhs, float rhs) { return lhs * rhs; }
DEFINE_FUNCTOR(ElemMulBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) { *dYdL = rhs; *dYdR = lhs; }

DEFINE_FUNCTOR(ElemDivForward, float, float lhs, float rhs) { return lhs / rhs; }
DEFINE_FUNCTOR(ElemDivBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) { *dYdL = 1.f / rhs; *dYdR = -lhs / (rhs * rhs); }

DEFINE_FUNCTOR(SoftMarginForward, float, float lhs, float rhs) { return log(1 + exp(-lhs * rhs)); }
DEFINE_FUNCTOR(SoftMarginBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) {
	*dYdL = -rhs / (1 + exp(lhs * rhs));
	*dYdR = -lhs / (1 + exp(lhs * rhs));
}

DEFINE_FUNCTOR(BinaryCrossEntropyForward, float, float lhs, float rhs) { return -rhs * log(lhs) - (1 - rhs) * log(1 - lhs); }
DEFINE_FUNCTOR(BinaryCrossEntropyBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) {
	*dYdL = (lhs - rhs) / (lhs - lhs * lhs);
	*dYdR = log(1 - lhs) - log(lhs);
}

DEFINE_FUNCTOR(BinaryClassificationAccuracyForward, float, float lhs, float rhs) { return (lhs > 0.5f) == (rhs > 0.5f); }

// Unary functors

DEFINE_FUNCTOR(ElemNegForward, float, float x) { return -x; }
DEFINE_FUNCTOR(ElemNegBackward, void, float x, float y, float *dYdX) { *dYdX = -1; }

DEFINE_FUNCTOR(SquareForward, float, float x) { return x * x; }
DEFINE_FUNCTOR(SquareBackward, void, float x, float y, float *dYdX) { *dYdX = 2.f * x; }

DEFINE_FUNCTOR(CubeForward, float, float x) { return x * x * x; }
DEFINE_FUNCTOR(CubeBackward, void, float x, float y, float *dYdX) { *dYdX = 3.f * x * x; }

DEFINE_FUNCTOR(ExpForward, float, float x) { return exp(x); }
DEFINE_FUNCTOR(ExpBackward, void, float x, float y, float *dYdX) { *dYdX = y; }

DEFINE_FUNCTOR(LogForward, float, float x) { return log(x); }
DEFINE_FUNCTOR(LogBackward, void, float x, float y, float *dYdX) { *dYdX = 1.f / x; }

DEFINE_FUNCTOR(AbsForward, float, float x) { return fabs(x); }
DEFINE_FUNCTOR(AbsBackward, void, float x, float y, float *dYdX) { *dYdX = (x > 0.f) ? 1.f : -1.f; }

DEFINE_FUNCTOR(SqrtForward, float, float x) { return sqrt(x); }
DEFINE_FUNCTOR(SqrtBackward, void, float x, float y, float *dYdX) { *dYdX = 0.5f / y; }

DEFINE_FUNCTOR(CbrtForward, float, float x) { return cbrt(x); }
DEFINE_FUNCTOR(CbrtBackward, void, float x, float y, float *dYdX) { *dYdX = (1.f / 3.f) / (y * y); }

DEFINE_FUNCTOR(SinForward, float, float x) { return sin(x); }
DEFINE_FUNCTOR(SinBackward, void, float x, float y, float *dYdX) { *dYdX = cos(x); }

DEFINE_FUNCTOR(CosForward, float, float x) { return cos(x); }
DEFINE_FUNCTOR(CosBackward, void, float x, float y, float *dYdX) { *dYdX = -sin(x); }

DEFINE_FUNCTOR(TanForward, float, float x) { return tan(x); }
DEFINE_FUNCTOR(TanBackward, void, float x, float y, float *dYdX) { *dYdX = 1.f / (cos(x) * cos(x)); }

DEFINE_FUNCTOR(AsinForward, float, float x) { return asin(x); }
DEFINE_FUNCTOR(AsinBackward, void, float x, float y, float *dYdX) { *dYdX = 1.f / sqrt(1.f - x * x); }

DEFINE_FUNCTOR(AcosForward, float, float x) { return acos(x); }
DEFINE_FUNCTOR(AcosBackward, void, float x, float y, float *dYdX) { *dYdX = -1.f / sqrt(1.f - x * x); }

DEFINE_FUNCTOR(AtanForward, float, float x) { return atan(x); }
DEFINE_FUNCTOR(AtanBackward, void, float x, float y, float *dYdX) { *dYdX = 1.f / (1.f + x * x); }

DEFINE_FUNCTOR(SinhForward, float, float x) { return sinh(x); }
DEFINE_FUNCTOR(SinhBackward, void, float x, float y, float *dYdX) { *dYdX = cosh(x); };

DEFINE_FUNCTOR(CoshForward, float, float x) { return cosh(x); }
DEFINE_FUNCTOR(CoshBackward, void, float x, float y, float *dYdX) { *dYdX = sinh(x); };

DEFINE_FUNCTOR(TanhForward, float, float x) { return tanh(x); }
DEFINE_FUNCTOR(TanhBackward, void, float x, float y, float *dYdX) { *dYdX = 1 - y * y; }

DEFINE_FUNCTOR(AsinhForward, float, float x) { return asinh(x); }
DEFINE_FUNCTOR(AsinhBackward, void, float x, float y, float *dYdX) { *dYdX = 1.f / sqrt(x * x + 1.f); }

DEFINE_FUNCTOR(AcoshForward, float, float x) { return acosh(x); }
DEFINE_FUNCTOR(AcoshBackward, void, float x, float y, float *dYdX) { *dYdX = 1.f / sqrt(x * x - 1.f); }

DEFINE_FUNCTOR(AtanhForward, float, float x) { return atanh(x); }
DEFINE_FUNCTOR(AtanhBackward, void, float x, float y, float *dYdX) { *dYdX = 1.f / (1.f - x * x); }

DEFINE_FUNCTOR(SigmoidForward, float, float x) { return 1 / (1 + exp(x)); }
DEFINE_FUNCTOR(SigmoidBackward, void, float x, float y, float *dYdX) { *dYdX = -y * y * exp(x); }

DEFINE_FUNCTOR(ReLUForward, float, float x) { return fmax(0.f, x); }
DEFINE_FUNCTOR(ReLUBackward, void, float x, float y, float *dYdX) { *dYdX = (x > 0.f) ? 1.f : 0.f; }

// Instantiation helpers
#define INSTANTIATE_BINARY(device_type, forward_func, backward_func) \
	template struct BinaryOpNodeFactory<device_type, forward_func, backward_func>;

#define INSTANTIATE_BINARY_OPS(device_type) \
	INSTANTIATE_BINARY(device_type, ElemAddForward, ElemAddBackward) \
	INSTANTIATE_BINARY(device_type, ElemSubForward, ElemSubBackward) \
	INSTANTIATE_BINARY(device_type, ElemMulForward, ElemMulBackward) \
	INSTANTIATE_BINARY(device_type, ElemDivForward, ElemDivBackward) \
	INSTANTIATE_BINARY(device_type, SoftMarginForward, SoftMarginBackward) \
	INSTANTIATE_BINARY(device_type, BinaryCrossEntropyForward, BinaryCrossEntropyBackward) \
	INSTANTIATE_BINARY(device_type, BinaryClassificationAccuracyForward, BinaryNoBackward)

#define INSTANTIATE_BINARY_LEFT_SCALAR(device_type, forward_func, backward_func) \
	template struct BinaryLeftScalarOpNodeFactory<device_type, forward_func, backward_func>;

#define INSTANTIATE_BINARY_LEFT_SCALAR_OPS(device_type) \
	INSTANTIATE_BINARY_LEFT_SCALAR(device_type, ElemAddForward, ElemAddBackward) \
	INSTANTIATE_BINARY_LEFT_SCALAR(device_type, ElemSubForward, ElemSubBackward) \
	INSTANTIATE_BINARY_LEFT_SCALAR(device_type, ElemMulForward, ElemMulBackward) \
	INSTANTIATE_BINARY_LEFT_SCALAR(device_type, ElemDivForward, ElemDivBackward)

#define INSTANTIATE_BINARY_RIGHT_SCALAR(device_type, forward_func, backward_func) \
	template struct BinaryRightScalarOpNodeFactory<device_type, forward_func, backward_func>;

#define INSTANTIATE_BINARY_RIGHT_SCALAR_OPS(device_type) \
	INSTANTIATE_BINARY_RIGHT_SCALAR(device_type, ElemAddForward, ElemAddBackward) \
	INSTANTIATE_BINARY_RIGHT_SCALAR(device_type, ElemSubForward, ElemSubBackward) \
	INSTANTIATE_BINARY_RIGHT_SCALAR(device_type, ElemMulForward, ElemMulBackward) \
	INSTANTIATE_BINARY_RIGHT_SCALAR(device_type, ElemDivForward, ElemDivBackward)

#define INSTANTIATE_UNARY(device_type, forward_func, backward_func) \
	template struct UnaryOpNodeFactory<device_type, forward_func, backward_func>;

#define INSTANTIATE_UNARY_OPS(device_type) \
	INSTANTIATE_UNARY(device_type, ElemNegForward, ElemNegBackward) \
	INSTANTIATE_UNARY(device_type, SquareForward, SquareBackward) \
	INSTANTIATE_UNARY(device_type, CubeForward, CubeBackward) \
	INSTANTIATE_UNARY(device_type, ExpForward, ExpBackward) \
	INSTANTIATE_UNARY(device_type, LogForward, LogBackward) \
	INSTANTIATE_UNARY(device_type, AbsForward, AbsBackward) \
	INSTANTIATE_UNARY(device_type, SqrtForward, SqrtBackward) \
	INSTANTIATE_UNARY(device_type, CbrtForward, CbrtBackward) \
	INSTANTIATE_UNARY(device_type, SinForward, SinBackward) \
	INSTANTIATE_UNARY(device_type, CosForward, CosBackward) \
	INSTANTIATE_UNARY(device_type, TanForward, TanBackward) \
	INSTANTIATE_UNARY(device_type, AsinForward, AsinBackward) \
	INSTANTIATE_UNARY(device_type, AcosForward, AcosBackward) \
	INSTANTIATE_UNARY(device_type, AtanForward, AtanBackward) \
	INSTANTIATE_UNARY(device_type, SinhForward, SinhBackward) \
	INSTANTIATE_UNARY(device_type, CoshForward, CoshBackward) \
	INSTANTIATE_UNARY(device_type, TanhForward, TanhBackward) \
	INSTANTIATE_UNARY(device_type, AsinhForward, AsinhBackward) \
	INSTANTIATE_UNARY(device_type, AcoshForward, AcoshBackward) \
	INSTANTIATE_UNARY(device_type, AtanhForward, AtanhBackward) \
	INSTANTIATE_UNARY(device_type, SigmoidForward, SigmoidBackward) \
	INSTANTIATE_UNARY(device_type, ReLUForward, ReLUBackward)

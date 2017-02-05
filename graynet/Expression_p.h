#pragma once

#include "Node.h"
#ifdef USE_CUDA
#include <host_defines.h>
#else
#define __device__
#define __host__
#endif

template<DeviceType DeviceType, typename ForwardFunc, typename BackwardFunc>
class BinaryOpNode : public Node {
public:
	BinaryOpNode(int lhs_node, int rhs_node);
};

template<DeviceType DeviceType, typename ForwardFunc, typename BackwardFunc>
class UnaryOpNode : public Node {
public:
	UnaryOpNode(int node);
};

template<typename Dummy, DeviceType DeviceType>
class SoftmaxNode : public Node {
public:
	SoftmaxNode(int node);
};

template<typename Dummy, DeviceType DeviceType>
class CrossEntropyNode : public Node {
public:
	CrossEntropyNode(Graph *graph, int node, const std::vector<int> &labels);
};

template<typename Dummy, DeviceType DeviceType>
class ClassificationAccuracyNode : public Node {
public:
	ClassificationAccuracyNode(Graph *graph, int node, const std::vector<int> &labels);
};

#define DEFINE_FUNCTOR(name, ret_type, ...) \
	struct name {	\
		inline __host__ __device__ ret_type operator()(__VA_ARGS__); \
	}; \
	inline __host__ __device__ ret_type name::operator()(__VA_ARGS__)

// Binary functors

DEFINE_FUNCTOR(ElemAddForward, float, float lhs, float rhs) { return lhs + rhs; }
DEFINE_FUNCTOR(ElemAddBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) { *dYdL = *dYdR = 1; }

DEFINE_FUNCTOR(ElemSubForward, float, float lhs, float rhs) { return lhs - rhs; }
DEFINE_FUNCTOR(ElemSubBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) { *dYdL = 1; *dYdR = -1; }

DEFINE_FUNCTOR(ElemMulForward, float, float lhs, float rhs) { return lhs * rhs; }
DEFINE_FUNCTOR(ElemMulBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) { *dYdL = rhs; *dYdR = lhs; }

DEFINE_FUNCTOR(ElemDivForward, float, float lhs, float rhs) { return lhs / rhs; }
DEFINE_FUNCTOR(ElemDivBackward, void, float lhs, float rhs, float y, float *dYdL, float *dYdR) { *dYdL = 1.f / rhs; *dYdR = -lhs / (rhs * rhs); }

// Unary functors

DEFINE_FUNCTOR(ElemNegForward, float, float x) { return -x; }
DEFINE_FUNCTOR(ElemNegBackward, void, float x, float y, float *dYdX) { *dYdX = -1; }

DEFINE_FUNCTOR(SqrForward, float, float x) { return x * x; }
DEFINE_FUNCTOR(SqrBackward, void, float x, float y, float *dYdX) { *dYdX = 2.f * x; }

DEFINE_FUNCTOR(CubeForward, float, float x) { return x * x * x; }
DEFINE_FUNCTOR(CubeBackward, void, float x, float y, float *dYdX) { *dYdX = 3.f * x * x; }

DEFINE_FUNCTOR(ExpForward, float, float x) { return exp(x); }
DEFINE_FUNCTOR(ExpBackward, void, float x, float y, float *dYdX) { *dYdX = y; }

DEFINE_FUNCTOR(SigmoidForward, float, float x) { return 1 / (1 + exp(x)); }
DEFINE_FUNCTOR(SigmoidBackward, void, float x, float y, float *dYdX) { *dYdX = -y * y * exp(x); }

DEFINE_FUNCTOR(TanhForward, float, float x) { return tanh(x); }
DEFINE_FUNCTOR(TanhBackward, void, float x, float y, float *dYdX) { *dYdX = 1 - y * y; }

DEFINE_FUNCTOR(ReLUForward, float, float x) { return fmax(0.f, x); }
DEFINE_FUNCTOR(ReLUBackward, void, float x, float y, float *dYdX) { *dYdX = (x > 0.f) ? 1.f : 0.f; }

// Instantiation helpers
#define INSTANTIATE_BINARY(device_type, forward_func, backward_func) \
	template class BinaryOpNode<device_type, forward_func, backward_func>;

#define INSTANTIATE_BINARY_OPS(device_type) \
	INSTANTIATE_BINARY(device_type, ElemAddForward, ElemAddBackward) \
	INSTANTIATE_BINARY(device_type, ElemSubForward, ElemSubBackward) \
	INSTANTIATE_BINARY(device_type, ElemMulForward, ElemMulBackward) \
	INSTANTIATE_BINARY(device_type, ElemDivForward, ElemDivBackward)

#define INSTANTIATE_UNARY(device_type, forward_func, backward_func) \
	template class UnaryOpNode<device_type, forward_func, backward_func>;

#define INSTANTIATE_UNARY_OPS(device_type) \
	INSTANTIATE_UNARY(device_type, ElemNegForward, ElemNegBackward) \
	INSTANTIATE_UNARY(device_type, SqrForward, SqrBackward) \
	INSTANTIATE_UNARY(device_type, CubeForward, CubeBackward) \
	INSTANTIATE_UNARY(device_type, ExpForward, ExpBackward) \
	INSTANTIATE_UNARY(device_type, SigmoidForward, SigmoidBackward) \
	INSTANTIATE_UNARY(device_type, TanhForward, TanhBackward) \
	INSTANTIATE_UNARY(device_type, ReLUForward, ReLUBackward)

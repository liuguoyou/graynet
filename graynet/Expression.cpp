#include "Device.h"
#include "Expression.h"
#include "Expression_p.h"
#include "Graph.h"
#include "Node.h"
#include "Utils.h"

#include <array>
#include <cblas.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#endif

Expression::Expression() : graph_(nullptr), index_(0) {
}

int Expression::GetBatchSize() const {
	return graph_->GetNodeBatchSize(index_);
}

Shape Expression::GetShape() const {
	return graph_->GetNodeShape(index_);
}

bool Expression::IsDense() const {
	return !graph_->IsNodeOutputSparse(index_);
}

bool Expression::IsSparse() const {
	return graph_->IsNodeOutputSparse(index_);
}

Tensor Expression::Forward() const {
	return graph_->Forward(*this);
}

void Expression::Backward() const {
	return graph_->Backward(*this);
}

template<template<typename, DeviceType> typename FactoryType, typename... TArg>
static Expression CreateDeviceSpecificInputNode(Graph *graph, int batch_size, const Shape &output_shape, TArg&&... arg) {
	Node *node;
#ifdef USE_CUDA
	if (graph->GetDeviceType() == GPU)
		node = FactoryType<void, GPU>().Create(std::forward<TArg>(arg)...);
	else
#endif
		node = FactoryType<void, CPU>().Create(std::forward<TArg>(arg)...);
	return graph->AddNode(node, output_shape, false, batch_size);
}

template<template<typename, DeviceType> typename FactoryType, typename... TArg>
static Expression CreateDeviceSpecificNode(Graph *graph, const Shape &output_shape, TArg&&... arg) {
	Node *node;
#ifdef USE_CUDA
	if (graph->GetDeviceType() == GPU)
		node = FactoryType<void, GPU>().Create(std::forward<TArg>(arg)...);
	else
#endif
		node = FactoryType<void, CPU>().Create(std::forward<TArg>(arg)...);
	return graph->AddNode(node, output_shape);
}

class InputNode : public Node {
public:
	InputNode(Graph *graph, int batch_size, const Shape &shape, const float *data)
		: Node{}, batch_size_(batch_size), shape_(shape) {
		int size = batch_size * shape.GetSize() * sizeof(float);
		data_ = (float *)PinMemory(graph->GetDevice(), data, size);
	}

	virtual void FreeMemory(Device *device) override {
		device->FreeMemoryPinned(data_);
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		float *y_data = y->GetData();
		int size = batch_size_ * shape_.GetSize() * sizeof(float);
		CopyMemoryHostToDeviceAsync(graph->GetDevice(), y_data, data_, size);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		// empty
	}

private:
	int batch_size_;
	Shape shape_;
	float *data_;
};

Expression Input(Graph *graph, const Shape &shape, const float *data) {
	return graph->AddNode(new InputNode(graph, 1, shape, data), shape, false, 1);
}

Expression BatchInput(Graph *graph, int batch_size, const Shape &shape, const float *data) {
	return graph->AddNode(new InputNode(graph, batch_size, shape, data), shape, false, batch_size);
}

class SparseInputNode : public Node {
public:
	SparseInputNode(Graph *graph, int batch_size, const Shape &shape, int nonzero_count,
		const float *sparse_data, const int *batch_indices, const int *indices)
		: Node{}, batch_size_(batch_size), shape_(shape), nonzero_count_(nonzero_count) {
		if (shape.GetRank() != 1)
			REPORT_ERROR("Shape of sparse input must be 1D.");
		sparse_data_ = (float *)PinMemory(graph->GetDevice(), sparse_data, nonzero_count * sizeof(float));
		batch_indices_ = (int *)PinMemory(graph->GetDevice(), batch_indices, (batch_size + 1) * sizeof(int));
		indices_ = (int *)PinMemory(graph->GetDevice(), indices, nonzero_count * sizeof(int));
	}

	virtual int GetFlags() const override {
		return NoAllocateForwardOutput;
	}

	virtual void FreeMemory(Device *device) override {
		device->FreeMemoryPinned(sparse_data_);
		device->FreeMemoryPinned(batch_indices_);
		device->FreeMemoryPinned(indices_);
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		Device *device = graph->GetDevice();
		float *sparse_data = (float*)device->AllocateMemory(nonzero_count_ * sizeof(float));
		int *batch_indices = (int*)device->AllocateMemory((batch_size_ + 1) * sizeof(int));
		int *indices = (int*)device->AllocateMemory(nonzero_count_ * sizeof(int));
		CopyMemoryHostToDeviceAsync(device, sparse_data, sparse_data_, nonzero_count_ * sizeof(float));
		CopyMemoryHostToDeviceAsync(device, batch_indices, batch_indices_, (batch_size_ + 1) * sizeof(int));
		CopyMemoryHostToDeviceAsync(device, indices, indices_, nonzero_count_ * sizeof(int));
		*y = Tensor(graph->GetDeviceType(), batch_size_, shape_, nonzero_count_, sparse_data, batch_indices, indices);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		// empty
	}

private:
	int batch_size_;
	Shape shape_;
	int nonzero_count_;
	float *sparse_data_;
	int *batch_indices_;
	int *indices_;
};

Expression BatchSparseVectorInput(Graph *graph, int batch_size, const Shape &shape,
	int nonzero_count, const float *sparse_data, const int *batch_indices, const int *indices) {
	return graph->AddNode(new SparseInputNode(graph, batch_size, shape, nonzero_count,
		sparse_data, batch_indices, indices), shape, true, batch_size);
}

class LookupNodeCPU : public Node {
public:
	LookupNodeCPU(Graph *graph, int embeddings, int batch_size, const Shape &shape, const int *indices)
		: Node{ embeddings }, batch_size_(batch_size), shape_(shape) {
		int size = batch_size * shape.GetSize();
		indices_ = (int*)PinMemory(graph->GetDevice(), indices, size * sizeof(int));
	}

	virtual void FreeMemory(Device *device) override {
		device->FreeMemoryPinned(indices_);
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *emb_data = x[0]->GetData();
		int count = batch_size_ * shape_.GetSize();
		int emb_size = x[0]->GetShape().GetDim(1);
		float *y_data = y->GetData();
		for (int i = 0; i < count; i++) {
			int index = indices_[i];
			memcpy(y_data, &emb_data[emb_size * index], emb_size * sizeof(float));
			y_data += emb_size;
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int count = batch_size_ * shape_.GetSize();
		int emb_size = x[0]->GetShape().GetDim(1);
		for (int i = 0; i < count; i++) {
			int index = indices_[i];
			cblas_saxpy(emb_size, 1.f, &dEdY_data[emb_size * i], 1,
				&dEdX_data[emb_size * index], 1);
		}
	}

private:
	int batch_size_;
	Shape shape_;
	int *indices_;
};

template<typename Dummy>
struct LookupNodeFactory<Dummy, CPU> {
	Node *Create(Graph *graph, int embeddings, int batch_size, const Shape &shape, const int *indices) {
		return new LookupNodeCPU(graph, embeddings, batch_size, shape, indices);
	}
};

Expression Lookup(const Expression &embeddings, const Shape &shape, const int *indices) {
	return BatchLookup(embeddings, 1, shape, indices);
}

Expression BatchLookup(const Expression &embeddings, int batch_size, const Shape &shape, const int *indices) {
	Graph *graph = embeddings.GetGraph();
	if (embeddings.GetBatchSize() != 1)
		REPORT_ERROR("Embedding table must be of batch size 1.");
	if (embeddings.GetShape().GetRank() != 2)
		REPORT_ERROR("Embedding table must be of rank 2.");
	int emb_size = embeddings.GetShape().GetDim(1);
	Shape output_shape = shape;
	output_shape.PushDim(emb_size);
	return CreateDeviceSpecificInputNode<LookupNodeFactory>(graph, batch_size, output_shape,
		graph, embeddings.GetNodeIndex(), batch_size, shape, indices);
}

namespace {
	template<int N, int I, int M, typename TransformFunc>
	struct TransformKernel {
		void Transform(TransformFunc &transform_func, const int *dims,
			const std::array<const int *, M> &strides, const std::array<int, M> &bases) {
			std::array<int, M> cur = bases;
			for (int j = 0; j < dims[I]; j++) {
				TransformKernel<N, I + 1, M, TransformFunc>().Transform(transform_func, dims, strides, cur);
				for (int k = 0; k < M; k++)
					cur[k] += strides[k][I];
			}
		}
	};

	template<int N, int M, typename TransformFunc>
	struct TransformKernel<N, N, M, TransformFunc> {
		void Transform(TransformFunc &transform_func, const int *dims,
			const std::array<const int *, M> &strides, const std::array<int, M> &bases) {
			apply(transform_func, bases);
		}
	};
}

template<int M, typename TransformFunc>
static void Transform(TransformFunc transform_func, int N, const int *dims,
	const std::array<const int *, M> &strides) {
	std::array<int, M> bases{};
	switch (N) {
	case 1: TransformKernel<1, 0, M, TransformFunc>().Transform(transform_func, dims, strides, bases); break;
	case 2: TransformKernel<2, 0, M, TransformFunc>().Transform(transform_func, dims, strides, bases); break;
	case 3: TransformKernel<3, 0, M, TransformFunc>().Transform(transform_func, dims, strides, bases); break;
	case 4: TransformKernel<4, 0, M, TransformFunc>().Transform(transform_func, dims, strides, bases); break;
	case 5: TransformKernel<5, 0, M, TransformFunc>().Transform(transform_func, dims, strides, bases); break;
	case 6: TransformKernel<6, 0, M, TransformFunc>().Transform(transform_func, dims, strides, bases); break;
	case 7: TransformKernel<7, 0, M, TransformFunc>().Transform(transform_func, dims, strides, bases); break;
	case 8: TransformKernel<8, 0, M, TransformFunc>().Transform(transform_func, dims, strides, bases); break;
	default:
		static_assert(8 == kMaxTensorDim + 1, "");
		DEBUG_BREAK();
	}
}

template<typename ForwardFunc, typename BackwardFunc>
class BinaryOpNodeCPU : public Node {
public:
	BinaryOpNodeCPU(int lhs_node, int rhs_node) : Node{ lhs_node, rhs_node } {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		float *y_data = y->GetData();
		int lhs_strides[kMaxTensorDim + 1], rhs_strides[kMaxTensorDim + 1];
		int y_strides[kMaxTensorDim + 1];
		GetTensorStrides(x[0], lhs_strides);
		GetTensorStrides(x[1], rhs_strides);
		GetTensorStrides(y, y_strides);
		int ndims = y->GetShape().GetRank() + 1;
		int dims[kMaxTensorDim + 1];
		GetTensorDims(y, dims);
		auto transform_func = [&](int lhs_index, int rhs_index, int y_index) {
			y_data[y_index] = ForwardFunc()(lhs_data[lhs_index], rhs_data[rhs_index]);
		};
		Transform<3>(transform_func, ndims, dims, { lhs_strides, rhs_strides, y_strides });
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value)
			REPORT_ERROR("Backward propagation is unsupported for this expression.");

		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdL_data = dEdX[0]->GetData(), *dEdR_data = dEdX[1]->GetData();
		int lhs_strides[kMaxTensorDim + 1], rhs_strides[kMaxTensorDim + 1];
		int y_strides[kMaxTensorDim + 1];
		GetTensorStrides(x[0], lhs_strides);
		GetTensorStrides(x[1], rhs_strides);
		GetTensorStrides(y, y_strides);
		int ndims = y->GetShape().GetRank() + 1;
		int dims[kMaxTensorDim + 1];
		GetTensorDims(y, dims);
		auto transform_func = [&](int lhs_index, int rhs_index, int y_index) {
			float dYdL, dYdR;
			BackwardFunc()(lhs_data[lhs_index], rhs_data[rhs_index], y_data[y_index], &dYdL, &dYdR);
			dEdL_data[lhs_index] += dYdL * dEdY_data[y_index];
			dEdR_data[rhs_index] += dYdR * dEdY_data[y_index];
		};
		Transform<3>(transform_func, ndims, dims, { lhs_strides, rhs_strides, y_strides });
	}
};

template<typename ForwardFunc, typename BackwardFunc>
struct BinaryOpNodeFactory<CPU, ForwardFunc, BackwardFunc> {
	Node *Create(int lhs_node, int rhs_node) {
		return new BinaryOpNodeCPU<ForwardFunc, BackwardFunc>(lhs_node, rhs_node);
	}
};

static Shape GetBroadcastingOutputShape(const Shape &lhs_shape, const Shape &rhs_shape) {
	// Broadcasting
	if (lhs_shape.GetRank() != rhs_shape.GetRank()) {
		REPORT_ERROR("Input operands have different ranks (%d and %d).",
			lhs_shape.GetRank(), rhs_shape.GetRank());
	}
	int ndims = lhs_shape.GetRank();
	Shape shape;
	for (int i = 0; i < ndims; i++) {
		if (lhs_shape.GetDim(i) == 1)
			shape.PushDim(rhs_shape.GetDim(i));
		else if (rhs_shape.GetDim(i) == 1)
			shape.PushDim(lhs_shape.GetDim(i));
		else if (lhs_shape.GetDim(i) == rhs_shape.GetDim(i))
			shape.PushDim(lhs_shape.GetDim(i));
		else {
			REPORT_ERROR("Incompatible size at dimension %d: %d and %d.",
				i, lhs_shape.GetDim(i), rhs_shape.GetDim(i));
		}
	}
	return shape;
}

template<typename ForwardFunc, typename BackwardFunc>
static Expression CreateBinaryOpNode(const Expression &lhs, const Expression &rhs) {
	Shape output_shape = GetBroadcastingOutputShape(lhs.GetShape(), rhs.GetShape());
	Graph *graph = lhs.GetGraph();
	Node *node;
#ifdef USE_CUDA
	if (graph->GetDeviceType() == GPU)
		node = BinaryOpNodeFactory<GPU, ForwardFunc, BackwardFunc>().Create(lhs.GetNodeIndex(), rhs.GetNodeIndex());
	else
#endif
		node = BinaryOpNodeFactory<CPU, ForwardFunc, BackwardFunc>().Create(lhs.GetNodeIndex(), rhs.GetNodeIndex());
	return graph->AddNode(node, output_shape);
}

Expression operator+(const Expression &lhs, const Expression &rhs) {
	Graph *graph = lhs.GetGraph();
	return CreateBinaryOpNode<ElemAddForward, ElemAddBackward>(lhs, rhs);
}

Expression &operator+=(Expression &lhs, const Expression &rhs) {
	lhs = lhs + rhs;
	return lhs;
}

Expression &operator+=(Expression &lhs, float rhs) {
	lhs = lhs + rhs;
	return lhs;
}

Expression operator-(const Expression &lhs, const Expression &rhs) {
	return CreateBinaryOpNode<ElemSubForward, ElemSubBackward>(lhs, rhs);
}

Expression &operator-=(Expression &lhs, const Expression &rhs) {
	lhs = lhs - rhs;
	return lhs;
}

Expression &operator-=(Expression &lhs, float rhs) {
	lhs = lhs - rhs;
	return lhs;
}

Expression operator*(const Expression &lhs, const Expression &rhs) {
	return CreateBinaryOpNode<ElemMulForward, ElemMulBackward>(lhs, rhs);
}

Expression &operator*=(Expression &lhs, const Expression &rhs) {
	lhs = lhs * rhs;
	return lhs;
}

Expression &operator*=(Expression &lhs, float rhs) {
	lhs = lhs * rhs;
	return lhs;
}

Expression operator/(const Expression &lhs, const Expression &rhs) {
	return CreateBinaryOpNode<ElemDivForward, ElemDivBackward>(lhs, rhs);
}

Expression &operator/=(Expression &lhs, const Expression &rhs) {
	lhs = lhs / rhs;
	return lhs;
}

Expression &operator/=(Expression &lhs, float rhs) {
	lhs = lhs / rhs;
	return lhs;
}

template<typename ForwardFunc, typename BackwardFunc>
class BinaryLeftScalarOpNodeCPU : public Node {
public:
	BinaryLeftScalarOpNodeCPU(float lhs_scalar, int rhs_node) : Node{ rhs_node }, lhs_scalar_(lhs_scalar) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *rhs_data = x[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		float *y_data = y->GetData();
		for (int i = 0; i < size; i++)
			y_data[i] = ForwardFunc()(lhs_scalar_, rhs_data[i]);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value)
			REPORT_ERROR("Backward propagation is unsupported for this expression.");

		const float *rhs_data = x[0]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdR_data = dEdX[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		for (int i = 0; i < size; i++) {
			float dYdL, dYdR;
			BackwardFunc()(lhs_scalar_, rhs_data[i], y_data[i], &dYdL, &dYdR);
			dEdR_data[i] += dYdR * dEdY_data[i];
		}
	}

private:
	float lhs_scalar_;
};

template<typename ForwardFunc, typename BackwardFunc>
struct BinaryLeftScalarOpNodeFactory<CPU, ForwardFunc, BackwardFunc> {
	Node *Create(float lhs_scalar, int rhs_node) {
		return new BinaryLeftScalarOpNodeCPU<ForwardFunc, BackwardFunc>(lhs_scalar, rhs_node);
	}
};

template<typename ForwardFunc, typename BackwardFunc>
static Expression CreateBinaryLeftScalarOpNode(float lhs_scalar, const Expression &rhs) {
	Graph *graph = rhs.GetGraph();
	Node *node;
#ifdef USE_CUDA
	if (graph->GetDeviceType() == GPU)
		node = BinaryLeftScalarOpNodeFactory<GPU, ForwardFunc, BackwardFunc>().Create(lhs_scalar, rhs.GetNodeIndex());
	else
#endif
		node = BinaryLeftScalarOpNodeFactory<CPU, ForwardFunc, BackwardFunc>().Create(lhs_scalar, rhs.GetNodeIndex());
	return graph->AddNode(node, rhs.GetShape());
}

Expression operator+(float lhs, const Expression &rhs) {
	return CreateBinaryLeftScalarOpNode<ElemAddForward, ElemAddBackward>(lhs, rhs);
}

Expression operator-(float lhs, const Expression &rhs) {
	return CreateBinaryLeftScalarOpNode<ElemSubForward, ElemSubBackward>(lhs, rhs);
}

Expression operator*(float lhs, const Expression &rhs) {
	return CreateBinaryLeftScalarOpNode<ElemMulForward, ElemMulBackward>(lhs, rhs);
}

Expression operator/(float lhs, const Expression &rhs) {
	return CreateBinaryLeftScalarOpNode<ElemDivForward, ElemDivBackward>(lhs, rhs);
}

template<typename ForwardFunc, typename BackwardFunc>
class BinaryRightScalarOpNodeCPU : public Node {
public:
	BinaryRightScalarOpNodeCPU(int lhs_node, float rhs_scalar) : Node{ lhs_node }, rhs_scalar_(rhs_scalar) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *lhs_data = x[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		float *y_data = y->GetData();
		for (int i = 0; i < size; i++)
			y_data[i] = ForwardFunc()(lhs_data[i], rhs_scalar_);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value)
			REPORT_ERROR("Backward propagation is unsupported for this expression.");

		const float *lhs_data = x[0]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdL_data = dEdX[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		for (int i = 0; i < size; i++) {
			float dYdL, dYdR;
			BackwardFunc()(lhs_data[i], rhs_scalar_, y_data[i], &dYdL, &dYdR);
			dEdL_data[i] += dYdL * dEdY_data[i];
		}
	}

private:
	float rhs_scalar_;
};

template<typename ForwardFunc, typename BackwardFunc>
struct BinaryRightScalarOpNodeFactory<CPU, ForwardFunc, BackwardFunc> {
	Node *Create(int lhs_node, float rhs_scalar) {
		return new BinaryRightScalarOpNodeCPU<ForwardFunc, BackwardFunc>(lhs_node, rhs_scalar);
	}
};

template<typename ForwardFunc, typename BackwardFunc>
static Expression CreateBinaryRightScalarOpNode(const Expression &lhs, float rhs_scalar) {
	Graph *graph = lhs.GetGraph();
	Node *node;
#ifdef USE_CUDA
	if (graph->GetDeviceType() == GPU)
		node = BinaryRightScalarOpNodeFactory<GPU, ForwardFunc, BackwardFunc>().Create(lhs.GetNodeIndex(), rhs_scalar);
	else
#endif
		node = BinaryRightScalarOpNodeFactory<CPU, ForwardFunc, BackwardFunc>().Create(lhs.GetNodeIndex(), rhs_scalar);
	return graph->AddNode(node, lhs.GetShape());
}

Expression operator+(const Expression &lhs, float rhs) {
	return CreateBinaryRightScalarOpNode<ElemAddForward, ElemAddBackward>(lhs, rhs);
}

Expression operator-(const Expression &lhs, float rhs) {
	return CreateBinaryRightScalarOpNode<ElemSubForward, ElemSubBackward>(lhs, rhs);
}

Expression operator*(const Expression &lhs, float rhs) {
	return CreateBinaryRightScalarOpNode<ElemMulForward, ElemMulBackward>(lhs, rhs);
}

Expression operator/(const Expression &lhs, float rhs) {
	return CreateBinaryRightScalarOpNode<ElemDivForward, ElemDivBackward>(lhs, rhs);
}

template<typename ForwardFunc, typename BackwardFunc>
class UnaryOpNodeCPU : public Node {
public:
	UnaryOpNodeCPU(int node) : Node{ node } {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *x_data = x[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		float *y_data = y->GetData();
		for (int i = 0; i < size; i++)
			y_data[i] = ForwardFunc()(x_data[i]);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *x_data = x[0]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		for (int i = 0; i < size; i++) {
			float dYdX;
			BackwardFunc()(x_data[i], y_data[i], &dYdX);
			dEdX_data[i] += dYdX * dEdY_data[i];
		}
	}
};

template<typename ForwardFunc, typename BackwardFunc>
struct UnaryOpNodeFactory<CPU, ForwardFunc, BackwardFunc> {
	Node *Create(int node) {
		return new UnaryOpNodeCPU<ForwardFunc, BackwardFunc>(node);
	}
};

template<typename ForwardFunc, typename BackwardFunc>
static Expression CreateUnaryOpNode(const Expression &x) {
	Graph *graph = x.GetGraph();
	Node *node;
#ifdef USE_CUDA
	if (graph->GetDeviceType() == GPU)
		node = UnaryOpNodeFactory<GPU, ForwardFunc, BackwardFunc>().Create(x.GetNodeIndex());
	else
#endif
		node = UnaryOpNodeFactory<CPU, ForwardFunc, BackwardFunc>().Create(x.GetNodeIndex());
	return graph->AddNode(node, x.GetShape());
}

Expression operator-(const Expression &x) {
	return CreateUnaryOpNode<ElemNegForward, ElemNegBackward>(x);
}

Expression Square(const Expression &x) {
	return CreateUnaryOpNode<SquareForward, SquareBackward>(x);
}

Expression Cube(const Expression &x) {
	return CreateUnaryOpNode<CubeForward, CubeBackward>(x);
}

Expression Exp(const Expression &x) {
	return CreateUnaryOpNode<ExpForward, ExpBackward>(x);
}

Expression Log(const Expression &x) {
	return CreateUnaryOpNode<LogForward, LogBackward>(x);
}

Expression Abs(const Expression &x) {
	return CreateUnaryOpNode<AbsForward, AbsBackward>(x);
}

Expression Sqrt(const Expression &x) {
	return CreateUnaryOpNode<SqrtForward, SqrtBackward>(x);
}

Expression Cbrt(const Expression &x) {
	return CreateUnaryOpNode<CbrtForward, CbrtBackward>(x);
}

Expression Sin(const Expression &x) {
	return CreateUnaryOpNode<SinForward, SinBackward>(x);
}

Expression Cos(const Expression &x) {
	return CreateUnaryOpNode<CosForward, CosBackward>(x);
}

Expression Tan(const Expression &x) {
	return CreateUnaryOpNode<TanForward, TanBackward>(x);
}

Expression Asin(const Expression &x) {
	return CreateUnaryOpNode<AsinForward, AsinBackward>(x);
}

Expression Acos(const Expression &x) {
	return CreateUnaryOpNode<AcosForward, AcosBackward>(x);
}

Expression Atan(const Expression &x) {
	return CreateUnaryOpNode<AtanForward, AtanBackward>(x);
}

Expression Sinh(const Expression &x) {
	return CreateUnaryOpNode<SinhForward, SinhBackward>(x);
}

Expression Cosh(const Expression &x) {
	return CreateUnaryOpNode<CoshForward, CoshBackward>(x);
}

Expression Tanh(const Expression &x) {
	return CreateUnaryOpNode<TanhForward, TanhBackward>(x);
}

Expression Asinh(const Expression &x) {
	return CreateUnaryOpNode<AsinhForward, AsinhBackward>(x);
}

Expression Acosh(const Expression &x) {
	return CreateUnaryOpNode<AcoshForward, AcoshBackward>(x);
}

Expression Atanh(const Expression &x) {
	return CreateUnaryOpNode<AtanhForward, AtanhBackward>(x);
}

Expression Sigmoid(const Expression &x) {
	return CreateUnaryOpNode<SigmoidForward, SigmoidBackward>(x);
}

Expression ReLU(const Expression &x) {
	return CreateUnaryOpNode<ReLUForward, ReLUBackward>(x);
}

static void SGEMM(Device *device, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
	int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
	// Use column major
	int lda = (transA == CblasNoTrans) ? K : M;
	int ldb = (transB == CblasNoTrans) ? N : K;
	int ldc = N;
#ifdef USE_CUDA
	if (device->GetDeviceType() == GPU) {
		cublasOperation_t tA = (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t tB = (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CALL(cublasSgemm_v2(device->GetCuBLASHandle(), tB, tA,
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
	}
	else
#endif
		cblas_sgemm(CblasColMajor, transB, transA,
			N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
}

static void SGEMM(Device *device, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, CBLAS_TRANSPOSE transC,
	int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
	if (transC == CblasNoTrans)
		SGEMM(device, transA, transB, M, N, K, alpha, A, B, beta, C);
	else {
		// (A*B)^T = B^T * A^T
		CBLAS_TRANSPOSE tA = (transA == CblasNoTrans) ? CblasTrans : CblasNoTrans;
		CBLAS_TRANSPOSE tB = (transB == CblasNoTrans) ? CblasTrans : CblasNoTrans;
		SGEMM(device, tB, tA, N, M, K, alpha, B, A, beta, C);
	}
}

static void BatchSGEMM(Device *device, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
	int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C,
	int batchA, int batchB, int batchC) {
	int batch_size = (batchA > batchB) ? batchA : batchB;
	int strideA = (batchA == 1) ? 0 : M * K;
	int strideB = (batchB == 1) ? 0 : K * N;
	int strideC = (batchC == 1) ? 0 : M * N;
#ifdef USE_CUDA
	if (device->GetDeviceType() == GPU && strideC > 0) {
		// Use column major
		int lda = (transA == CblasNoTrans) ? K : M;
		int ldb = (transB == CblasNoTrans) ? N : K;
		int ldc = N;
		cublasOperation_t tA = (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t tB = (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CALL(cublasSgemmStridedBatched(device->GetCuBLASHandle(), tB, tA,
			N, M, K, &alpha, B, ldb, strideB, A, lda, strideA, &beta, C, ldc, strideC, batch_size));
	}
	else
#endif
	{
		for (int i = 0; i < batch_size; i++) {
			SGEMM(device, transA, transB, M, N, K,
				alpha, A + strideA * i, B + strideB * i, beta, C + strideC * i);
		}
	}
}

class MatMulNode : public Node {
public:
	MatMulNode(int lhs, int rhs) : Node{ lhs, rhs } {}

	void InferShapes(const Tensor *lhs, const Tensor *rhs, int *M, int *K, int *N,
		int *left_stack_size, int *right_stack_size, int *y_stack_size) const {
		const Shape &lhs_shape = lhs->GetShape();
		const Shape &rhs_shape = rhs->GetShape();
		*M = lhs_shape.GetRank() >= 2 ? lhs_shape.GetDim(lhs_shape.GetRank() - 2) : 1;
		*K = lhs_shape.GetDim(lhs_shape.GetRank() - 1);
		*N = rhs_shape.GetRank() >= 2 ? rhs_shape.GetDim(rhs_shape.GetRank() - 1) : 1;
		*left_stack_size = lhs->GetBatchSize() * lhs_shape.GetSizeRange(0, lhs_shape.GetRank() - 2);
		*right_stack_size = rhs->GetBatchSize() * rhs_shape.GetSizeRange(0, rhs_shape.GetRank() - 2);
		*y_stack_size = (*left_stack_size > *right_stack_size) ? *left_stack_size : *right_stack_size;
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		// y = L * R
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		int M, K, N;
		int left_stack_size, right_stack_size, y_stack_size;
		InferShapes(x[0], x[1], &M, &K, &N, &left_stack_size, &right_stack_size, &y_stack_size);
		float *y_data = y->GetData();
		if (left_stack_size == 1 && N == 1) {
			SGEMM(graph->GetDevice(), CblasNoTrans, CblasTrans, CblasTrans,
				M, N * right_stack_size, K, 1.f, lhs_data, rhs_data, 0.f, y_data);
		}
		else if (right_stack_size == 1) {
			SGEMM(graph->GetDevice(), CblasNoTrans, CblasNoTrans, CblasNoTrans,
				M * left_stack_size, N, K, 1.f, lhs_data, rhs_data, 0.f, y_data);
		}
		else {
			BatchSGEMM(graph->GetDevice(), CblasNoTrans, CblasNoTrans,
				M, N, K,
				1.f, lhs_data, rhs_data, 0.f, y_data,
				left_stack_size, right_stack_size, y_stack_size);
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		int M, K, N;
		int left_stack_size, right_stack_size, y_stack_size;
		InferShapes(x[0], x[1], &M, &K, &N, &left_stack_size, &right_stack_size, &y_stack_size);
		const float *dEdY_data = dEdY->GetData();
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		float *dEdL_data = dEdX[0]->GetData(), *dEdR_data = dEdX[1]->GetData();
		// dEdL += dEdY * R'
		// dEdR += L' * dEdY
		if (left_stack_size == 1 && N == 1) {
			SGEMM(graph->GetDevice(), CblasTrans, CblasNoTrans, CblasNoTrans,
				M, K, N * right_stack_size,
				1.f, dEdY_data, rhs_data, 1.f, dEdL_data);
			SGEMM(graph->GetDevice(), CblasTrans, CblasTrans, CblasTrans,
				K, N * right_stack_size, M,
				1.f, lhs_data, dEdY_data, 1.f, dEdR_data);
		}
		else if (right_stack_size == 1) {
			SGEMM(graph->GetDevice(), CblasNoTrans, CblasTrans, CblasNoTrans,
				M * left_stack_size, K, N,
				1.f, dEdY_data, rhs_data, 1.f, dEdL_data);
			SGEMM(graph->GetDevice(), CblasTrans, CblasNoTrans, CblasNoTrans,
				K, N, M * left_stack_size,
				1.f, lhs_data, dEdY_data, 1.f, dEdR_data);
		}
		else {
			BatchSGEMM(graph->GetDevice(), CblasNoTrans, CblasTrans,
				M, K, N,
				1.f, dEdY_data, rhs_data, 1.f, dEdL_data,
				y_stack_size, right_stack_size, left_stack_size);
			BatchSGEMM(graph->GetDevice(), CblasTrans, CblasNoTrans,
				K, N, M,
				1.f, lhs_data, dEdY_data, 1.f, dEdR_data,
				left_stack_size, y_stack_size, right_stack_size);
		}
	}
};

Expression MatMul(const Expression &lhs, const Expression &rhs) {
	/* Infer output shape. */
	const Shape &lhs_shape = lhs.GetShape(), &rhs_shape = rhs.GetShape();
	if (lhs_shape.GetRank() == 1 && rhs_shape.GetRank() == 1)
		REPORT_ERROR("Left and right operands are both vectors, use Dot() for now.");
	if (lhs_shape.GetRank() > 2 && rhs_shape.GetRank() > 2)
		REPORT_ERROR("MatMul() does not currently support the case when both inputs are stacks of matrices.");
	Shape output_shape;
	if (lhs_shape.GetRank() > 2) {
		for (int i = 0; i < lhs_shape.GetRank() - 2; i++)
			output_shape.PushDim(lhs_shape.GetDim(i));
	}
	else if (rhs_shape.GetRank() > 2) {
		for (int i = 0; i < rhs_shape.GetRank() - 2; i++)
			output_shape.PushDim(rhs_shape.GetDim(i));
	}
	if (lhs_shape.GetRank() == 1) {
		if (lhs_shape.GetDim(0) != rhs_shape.GetDim(rhs_shape.GetRank() - 2)) {
			REPORT_ERROR("Dimension mismatch for vector-matrix multiplication: (%d) * (%d, %d).",
				lhs_shape.GetDim(0),
				rhs_shape.GetDim(rhs_shape.GetRank() - 1),
				rhs_shape.GetDim(rhs_shape.GetRank() - 2));
		}
		output_shape.PushDim(rhs_shape.GetDim(rhs_shape.GetRank() - 1));
	}
	else if (rhs_shape.GetRank() == 1) {
		if (lhs_shape.GetDim(lhs_shape.GetRank() - 1) != rhs_shape.GetDim(0)) {
			REPORT_ERROR("Dimension mismatch for matrix-vector multiplication: (%d) * (%d, %d).",
				lhs_shape.GetDim(lhs_shape.GetRank() - 2),
				lhs_shape.GetDim(lhs_shape.GetRank() - 1),
				rhs_shape.GetDim(0));
		}
		output_shape.PushDim(lhs_shape.GetDim(lhs_shape.GetRank() - 2));
	}
	else {
		if (lhs_shape.GetDim(lhs_shape.GetRank() - 1) != rhs_shape.GetDim(rhs_shape.GetRank() - 2)) {
			REPORT_ERROR("Dimension mismatch for matrix-matrix multiplication: (%d, %d) * (%d, %d).",
				lhs_shape.GetDim(lhs_shape.GetRank() - 2),
				lhs_shape.GetDim(lhs_shape.GetRank() - 1),
				rhs_shape.GetDim(rhs_shape.GetRank() - 1),
				rhs_shape.GetDim(rhs_shape.GetRank() - 2));
		}
		output_shape.PushDim(lhs_shape.GetDim(lhs_shape.GetRank() - 2));
		output_shape.PushDim(rhs_shape.GetDim(rhs_shape.GetRank() - 1));
	}

	Graph *graph = lhs.GetGraph();
	return graph->AddNode(new MatMulNode(lhs.GetNodeIndex(), rhs.GetNodeIndex()), output_shape);
}

class SparseDotNodeCPU : public Node {
public:
	SparseDotNodeCPU(int lhs_node, int rhs_node) : Node{ lhs_node, rhs_node } {}

	virtual int GetFlags() const override {
		return NoAllocateBackwardOutput;
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *lhs_data = x[0]->GetSparseData();
		const int *row_indices = x[0]->GetSparseRowIndices();
		const int *column_indices = x[0]->GetSparseColumnIndices();
		int nnz = x[0]->GetNonZeroCount();
		const float *rhs_data = x[1]->GetData();
		float *y_data = y->GetData();
		int rows = y->GetBatchSize();
		
		graph->GetDevice()->ZeroMemory(y_data, rows * sizeof(float));
		for (int i = 0; i < rows; i++) {
			float sum = 0;
			for (int j = row_indices[i]; j < row_indices[i + 1]; j++)
				sum += lhs_data[j] * rhs_data[column_indices[j]];
			y_data[i] = sum;
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		// dEdL not implemented for now.
		AllocateClearTensor(graph, dEdX[1]);

		const int *row_indices = x[0]->GetSparseRowIndices();
		const int *column_indices = x[0]->GetSparseColumnIndices();
		int nnz = x[0]->GetNonZeroCount();
		const float *x_data = x[1]->GetData();
		const float *dEdY_data = dEdY->GetData();
		int rows = y->GetBatchSize();
		const float *lhs_data = x[0]->GetSparseData();
		float *dEdR_data = dEdX[1]->GetData();

		for (int i = 0; i < rows; i++)
			for (int j = row_indices[i]; j < row_indices[i + 1]; j++)
				dEdR_data[column_indices[j]] += lhs_data[j] * dEdY_data[i];
	}
};

template<typename Dummy>
struct SparseDotNodeFactory<Dummy, CPU> {
	Node *Create(int lhs_node, int rhs_node) {
		return new SparseDotNodeCPU(lhs_node, rhs_node);
	}
};

Expression Dot(const Expression &lhs, const Expression &rhs) {
	const Shape &lhs_shape = lhs.GetShape(), &rhs_shape = rhs.GetShape();
	if (lhs_shape.GetRank() != 1 || rhs_shape.GetRank() != 1)
		REPORT_ERROR("Dot only supports vector inputs.");
	if (lhs_shape.GetDim(0) != rhs_shape.GetDim(0))
		REPORT_ERROR("Length of dot operands mismatch.");
	Graph *graph = lhs.GetGraph();
	if (lhs.IsDense() && rhs.IsDense())
		return ReduceSum(lhs * rhs);
	else if (lhs.IsSparse() && rhs.IsDense()) {
		if (rhs.GetBatchSize() != 1)
			REPORT_ERROR("Batch size of dense vector must be 1.");
		return CreateDeviceSpecificNode<SparseDotNodeFactory>(graph, Shape(1), lhs.GetNodeIndex(), rhs.GetNodeIndex());
	}
	else if (lhs.IsDense() && rhs.IsSparse()) {
		if (lhs.GetBatchSize() != 1)
			REPORT_ERROR("Batch size of dense vector must be 1.");
		return CreateDeviceSpecificNode<SparseDotNodeFactory>(graph, Shape(1), rhs.GetNodeIndex(), lhs.GetNodeIndex());
	}
	else
		REPORT_ERROR("Sparse-sparse vector dot is unsupported.");
}

static Shape FilterForwardShape(const Shape &x_shape, const Shape &filter_shape,
	const Shape &strides, const Shape &padding, bool is_pooling) {
	int filter_window_offset = is_pooling ? 0 : 2;

	if (x_shape.GetRank() < 2)
		REPORT_ERROR("Input must have at least rank 2.");
	int dims = x_shape.GetRank() - 1;
	if (filter_shape.GetRank() != filter_window_offset + dims)
		REPORT_ERROR("Incompatible filter shape.");
	if (strides.GetRank() != dims)
		REPORT_ERROR("Incompatible strides.");
	int input_channels = x_shape.GetDim(0);
	int output_channels;
	if (is_pooling)
		output_channels = input_channels;
	else {
		if (filter_shape.GetDim(1) != input_channels)
			REPORT_ERROR("Incompatible input and filter shape.");
		output_channels = filter_shape.GetDim(0);
	}

	Shape ret_shape;
	ret_shape.PushDim(output_channels);
	for (int i = 0; i < dims; i++) {
		int input_size = x_shape.GetDim(1 + i);
		int filter_size = filter_shape.GetDim(filter_window_offset + i);
		int stride = strides.GetDim(i);
		int pad = padding.GetDim(i);
		int output_size = 1 + (input_size + pad * 2 - filter_size) / stride;
		ret_shape.PushDim(output_size);
	}
	return ret_shape;
}

class ConvolutionNode : public Node {
public:
	ConvolutionNode(int x, int filter, const Shape &strides, const Shape &padding):
		Node{ x, filter }, strides_(strides), padding_(padding) {
#ifdef USE_CUDA
		CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
		CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
		CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_));
		CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_));
#endif
	}

	virtual ~ConvolutionNode() {
#ifdef USE_CUDA
		CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
		CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc_));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc_));
#endif
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &x_shape = x[0]->GetShape();
		const Shape &filter_shape = x[1]->GetShape();
		const Shape &y_shape = y->GetShape();
		const float *x_data = x[0]->GetData();
		const float *filter_data = x[1]->GetData();
		float *y_data = y->GetData();
		int dims = y->GetShape().GetRank() - 1;

		if (graph->GetDeviceType() == CPU)
			REPORT_ERROR("Convolution is only implemented in GPU.");

#ifdef USE_CUDA
		int x_dims[CUDNN_DIM_MAX], y_dims[CUDNN_DIM_MAX];
		x_dims[0] = x[0]->GetBatchSize();
		y_dims[0] = y->GetBatchSize();
		for (int i = 0; i < dims + 1; i++) {
			x_dims[i + 1] = x_shape.GetDim(i);
			y_dims[i + 1] = y_shape.GetDim(i);
		}
		int x_strides[CUDNN_DIM_MAX], y_strides[CUDNN_DIM_MAX];
		x_strides[dims + 1] = 1;
		y_strides[dims + 1] = 1;
		for (int i = dims; i >= 0; i--) {
			x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
			y_strides[i] = y_dims[i + 1] * y_strides[i + 1];
		}

		CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_desc_, dims,
			padding_.data(), strides_.data(), Shape::One(dims).data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
		CUDNN_CALL(cudnnSetFilterNdDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			dims + 2, filter_shape.data()));
		CUDNN_CALL(cudnnSetTensorNdDescriptor(x_desc_, CUDNN_DATA_FLOAT,
			dims + 2, x_dims, x_strides));
		CUDNN_CALL(cudnnSetTensorNdDescriptor(y_desc_, CUDNN_DATA_FLOAT,
			dims + 2, y_dims, y_strides));

		// TODO: Use workspace for potential better performance
		cudnnConvolutionFwdAlgo_t fwd_algo;
		CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(graph->GetDevice()->GetCuDNNHandle(),
			x_desc_, filter_desc_, conv_desc_, y_desc_, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &fwd_algo));

		float alpha = 1.f, beta = 0.f;
		CUDNN_CALL(cudnnConvolutionForward(graph->GetDevice()->GetCuDNNHandle(),
			&alpha, x_desc_, x_data, filter_desc_, filter_data, conv_desc_,
			fwd_algo, nullptr, 0, &beta, y_desc_, y_data));
#endif
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *x_data = x[0]->GetData();
		const float *filter_data = x[1]->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		float *dEdF_data = dEdX[1]->GetData();

		if (graph->GetDeviceType() == CPU)
			REPORT_ERROR("Convolution is only implemented in GPU.");

#ifdef USE_CUDA
		float alpha = 1.f, beta = 1.f;

		cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
		CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(graph->GetDevice()->GetCuDNNHandle(),
			x_desc_, y_desc_, conv_desc_, filter_desc_, CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
			0, &bwd_filter_algo));
		CUDNN_CALL(cudnnConvolutionBackwardFilter(graph->GetDevice()->GetCuDNNHandle(),
			&alpha, x_desc_, x_data, y_desc_, dEdY_data, conv_desc_, bwd_filter_algo, nullptr, 0,
			&beta, filter_desc_, dEdF_data));

		cudnnDataType_t dataType;
		int nbDims;
		int dimA[CUDNN_DIM_MAX];
		int strideA[CUDNN_DIM_MAX];
		CUDNN_CALL(cudnnGetTensorNdDescriptor(y_desc_, 4, &dataType, &nbDims, dimA, strideA));
		
		cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
		CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(graph->GetDevice()->GetCuDNNHandle(),
			filter_desc_, y_desc_, conv_desc_, x_desc_, CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
			0, &bwd_data_algo));
		CUDNN_CALL(cudnnConvolutionBackwardData(graph->GetDevice()->GetCuDNNHandle(),
			&alpha, filter_desc_, filter_data, y_desc_, dEdY_data, conv_desc_, bwd_data_algo, nullptr, 0,
			&beta, x_desc_, dEdX_data));
#endif
	}

private:
	Shape strides_, padding_;
#ifdef USE_CUDA
	cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
	cudnnFilterDescriptor_t filter_desc_ = nullptr;
	cudnnTensorDescriptor_t x_desc_ = nullptr, y_desc_ = nullptr;
#endif
};

Expression Convolution(const Expression &x, const Expression &filter, const Shape &strides, const Shape &padding) {
	const Shape &x_shape = x.GetShape(), &filter_shape = filter.GetShape();
	Shape output_shape = FilterForwardShape(x_shape, filter_shape, strides, padding, false);

	Graph *graph = filter.GetGraph();
	return graph->AddNode(new ConvolutionNode(x.GetNodeIndex(), filter.GetNodeIndex(), strides, padding), output_shape);
}

namespace {
	template<int N, int I, typename EnumerateFunc>
	struct EnumerateTileKernel {
		FORCEINLINE void EnumerateTile(EnumerateFunc &enumerate_func, const std::array<int, N * 2 + 1> &input,
			const int *strides, int index) {
			int dim_start = input[I * 2 + 1];
			int dim_end = input[I * 2 + 2];
			for (int j = dim_start; j < dim_end; j++)
				EnumerateTileKernel<N, I + 1, EnumerateFunc>().EnumerateTile(
					enumerate_func, input, strides, index + strides[I + 1] * j);
		}
	};

	template<int N, typename EnumerateFunc>
	struct EnumerateTileKernel<N, N, EnumerateFunc> {
		FORCEINLINE void EnumerateTile(EnumerateFunc &enumerate_func, const std::array<int, N * 2 + 1> &input,
			const int *strides, int index) {
			enumerate_func(index);
		}
	};

	template<int N, typename EnumerateFunc>
	static FORCEINLINE void EnumerateTile(EnumerateFunc &enumerate_func, const std::array<int, N * 2 + 1> &input,
		const int *strides) {
		int index = strides[0] * input[0];
		EnumerateTileKernel<N, 0, EnumerateFunc>().EnumerateTile(enumerate_func, input, strides, index);
	}

	template<int N, int I, typename TransformFunc>
	struct ForEachTileKernel {
		FORCEINLINE void ForEachTile(TransformFunc &transform_func, const Shape &x_shape,
			const Shape &filter_shape, const Shape &strides, const Shape &padding, std::array<int, N * 2 + 1> &input) {
			int dim_start = -padding.GetDim(I);
			int dim_end = x_shape.GetDim(I + 1) + padding.GetDim(I) - filter_shape.GetDim(I) + 1;
			for (int j = dim_start; j < dim_end; j++) {
				input[I * 2 + 1] = j;
				input[I * 2 + 2] = j + filter_shape.GetDim(I);
				if (input[I * 2 + 1] < 0)
					input[I * 2 + 1] = 0;
				if (input[I * 2 + 2] > x_shape.GetDim(I + 1))
					input[I * 2 + 2] = x_shape.GetDim(I + 1);
				ForEachTileKernel<N, I + 1, TransformFunc>().ForEachTile(
					transform_func, x_shape, filter_shape, strides, padding, input);
			}
		}
	};

	template<int N, typename TransformFunc>
	struct ForEachTileKernel<N, N, TransformFunc> {
		FORCEINLINE void ForEachTile(TransformFunc &transform_func, const Shape &x_shape,
			const Shape &filter_shape, const Shape &strides, const Shape &padding, std::array<int, N * 2 + 1> &input) {
			transform_func(input);
		}
	};

	template<int N, typename TransformFunc>
	static void ForEachTile(TransformFunc transform_func, int batch_size, const Shape &x_shape,
		const Shape &filter_shape, const Shape &strides, const Shape &padding) {
		std::array<int, N * 2 + 1> input;
		for (int i = 0; i < batch_size; i++) {
			input[0] = i;
			ForEachTileKernel<N, 0, TransformFunc>().ForEachTile(
				transform_func, x_shape,
				filter_shape, strides, padding, input);
		}
	}

	template<template<int> typename TransformFunc, typename... T>
	void ForEachTile(int ndims, int batch_size, const Shape &x_shape, const Shape &filter_shape,
		const Shape &strides, const Shape &padding, T&&... transform_func_arg) {
		switch (ndims) {
		case 1: ForEachTile<1>(TransformFunc<1>{std::forward<T>(transform_func_arg)...},
			batch_size, x_shape, filter_shape, strides, padding); break;
		case 2: ForEachTile<2>(TransformFunc<2>{std::forward<T>(transform_func_arg)...},
			batch_size, x_shape, filter_shape, strides, padding); break;
		case 3: ForEachTile<3>(TransformFunc<3>{std::forward<T>(transform_func_arg)...},
			batch_size, x_shape, filter_shape, strides, padding); break;
		case 4: ForEachTile<4>(TransformFunc<4>{std::forward<T>(transform_func_arg)...},
			batch_size, x_shape, filter_shape, strides, padding); break;
		case 5: ForEachTile<5>(TransformFunc<5>{std::forward<T>(transform_func_arg)...},
			batch_size, x_shape, filter_shape, strides, padding); break;
		case 6: ForEachTile<6>(TransformFunc<6>{std::forward<T>(transform_func_arg)...},
			batch_size, x_shape, filter_shape, strides, padding); break;
		case 7: ForEachTile<7>(TransformFunc<7>{std::forward<T>(transform_func_arg)...},
			batch_size, x_shape, filter_shape, strides, padding); break;
		default:
			static_assert(7 == kMaxTensorDim, "");
			REPORT_ERROR("Unsupported dimension count: %d", ndims);
		}
	}

	template<int N>
	struct MaxPoolingForwardKernel {
		const int *x_strides;
		const float *x;
		float *y;

		FORCEINLINE void operator()(const std::array<int, N * 2 + 1> &input) {
			float cur = -std::numeric_limits<float>::infinity();
			auto enumerate_func = [&](int index) {
				if (x[index] > cur)
					cur = x[index];
			};
			EnumerateTile<N>(enumerate_func, input, x_strides);
			*y++ = cur;
		}
	};

	template<int N>
	struct MaxPoolingBackwardKernel {
		const int *x_strides;
		const float *x, *y;
		const float *dEdY;
		float *dEdX;

		FORCEINLINE void operator()(const std::array<int, N * 2 + 1> &input) {
			auto enumerate_func = [&](int index) {
				if (x[index] == *y)
					dEdX[index] += *dEdY;
			};
			EnumerateTile<N>(enumerate_func, input, x_strides);
			dEdY++;
			y++;
		}
	};

	template<int N>
	struct AvgPoolingForwardKernel {
		const int *x_strides;
		const float *x;
		float *y;

		FORCEINLINE void operator()(const std::array<int, N * 2 + 1> &input) {
			float cur = 0;
			int cnt = 0;
			auto enumerate_func = [&](int index) {
				cur += x[index];
				cnt++;
			};
			EnumerateTile<N>(enumerate_func, input, x_strides);
			*y++ = (cnt == 0) ? 0.f : cur / cnt;
		}
	};

	template<int N>
	struct AvgPoolingBackwardKernel {
		const int *x_strides;
		const float *dEdY;
		float *dEdX;

		FORCEINLINE void operator()(const std::array<int, N * 2 + 1> &input) {
			int cnt = 0;
			auto enumerate_func1 = [&](int index) {
				cnt++;
			};
			EnumerateTile<N>(enumerate_func1, input, x_strides);
			float m = 1.f / cnt;
			auto enumerate_func2 = [&](int index) {
				dEdX[index] += *dEdY * m;
			};
			EnumerateTile<N>(enumerate_func2, input, x_strides);
			dEdY++;
		}
	};
}

class PoolingNodeCPU : public Node {
public:
	PoolingNodeCPU(int node, const Shape &filter_shape, const Shape &strides, const Shape &padding, PoolingMode mode)
		: Node{ node }, filter_shape_(filter_shape), strides_(strides), padding_(padding), mode_(mode) {
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &x_shape = x[0]->GetShape();
		const Shape &y_shape = y->GetShape();
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		int batch_size = y->GetBatchSize() * y_shape.GetDim(0);
		int ndims = y_shape.GetRank() - 1;
		int x_strides[kMaxTensorDim + 1];
		GetTensorStrides(x[0], x_strides);
		if (x_strides[1] == 0)
			x_strides[1] = x_strides[0];

		switch (mode_) {
		case PoolingMode::MaxPooling:
			ForEachTile<MaxPoolingForwardKernel>(ndims, batch_size, x_shape, filter_shape_, strides_, padding_,
				&x_strides[1], x_data, y_data);
			break;

		case PoolingMode::AvgPooling:
			ForEachTile<AvgPoolingForwardKernel>(ndims, batch_size, x_shape, filter_shape_, strides_, padding_,
				&x_strides[1], x_data, y_data);
			break;

		case PoolingMode::AvgPoolingWithPadding:
			//ForEachTile<AvgPoolingWithPaddingForwardKernel>(ndims, batch_size, x_shape, filter_shape_, strides_, padding_,
			//	&x_strides[1], x_data, y_data);
			REPORT_ERROR("AvgPoolingWithPadding not implemented.");
			break;

		default:
			DEBUG_BREAK();
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const Shape &x_shape = x[0]->GetShape();
		const Shape &y_shape = y->GetShape();
		const float *x_data = x[0]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int batch_size = y->GetBatchSize() * y_shape.GetDim(0);
		int ndims = y->GetShape().GetRank() - 1;
		int x_strides[kMaxTensorDim + 1];
		GetTensorStrides(x[0], x_strides);
		if (x_strides[1] == 0)
			x_strides[1] = x_strides[0];

		switch (mode_) {
		case PoolingMode::MaxPooling:
			ForEachTile<MaxPoolingBackwardKernel>(ndims, batch_size, x_shape, filter_shape_, strides_, padding_,
				&x_strides[1], x_data, y_data, dEdY_data, dEdX_data);
			break;

		case PoolingMode::AvgPooling:
			ForEachTile<AvgPoolingBackwardKernel>(ndims, batch_size, x_shape, filter_shape_, strides_, padding_,
				&x_strides[1], dEdY_data, dEdX_data);
			break;

		case PoolingMode::AvgPoolingWithPadding:
			//ForEachTile<AvgPoolingWithPaddingBackwardKernel>(ndims, batch_size, x_shape, filter_shape_, strides_, padding_,
			//	&x_strides[1], dEdY_data, dEdX_data);
			REPORT_ERROR("AvgPoolingWithPadding not implemented.");
			break;

		default:
			DEBUG_BREAK();
		}
	}

private:
	Shape filter_shape_, strides_, padding_;
	PoolingMode mode_;
};

template<typename Dummy>
struct PoolingNodeFactory<Dummy, CPU> {
	Node *Create(int node, const Shape &filter_shape, const Shape &strides, const Shape &padding, PoolingMode mode) {
		return new PoolingNodeCPU(node, filter_shape, strides, padding, mode);
	}
};

Expression MaxPooling(const Expression &x, const Shape &filter_shape, const Shape &strides, const Shape &padding) {
	Shape output_shape = FilterForwardShape(x.GetShape(), filter_shape, strides, padding, true);
	Graph *graph = x.GetGraph();
	return CreateDeviceSpecificNode<PoolingNodeFactory>(graph, output_shape, x.GetNodeIndex(), filter_shape, strides, padding, PoolingMode::MaxPooling);
}

Expression AvgPooling(const Expression &x, const Shape &filter_shape, const Shape &strides, const Shape &padding) {
	Shape output_shape = FilterForwardShape(x.GetShape(), filter_shape, strides, padding, true);
	Graph *graph = x.GetGraph();
	return CreateDeviceSpecificNode<PoolingNodeFactory>(graph, output_shape, x.GetNodeIndex(), filter_shape, strides, padding, PoolingMode::AvgPooling);
}

class ReshapeNode : public Node {
public:
	ReshapeNode(int node, const Shape &shape) : Node{ node }, shape_(shape) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		int size = y->GetBatchSize() * y->GetShape().GetSize() * sizeof(float);
		// TODO: Cannot use a view because it will cause double free in Graph::Clear().
		graph->GetDevice()->CopyMemory(y->GetData(), x[0]->GetData(), size);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = (float*)dEdY->GetData();
		float *dEdX_data = (float*)dEdX[0]->GetData();
		int size = dEdX[0]->GetBatchSize() * dEdX[0]->GetShape().GetSize();
		float alpha = 1.f;
#ifdef USE_CUDA
		if (graph->GetDeviceType() == GPU) {
			cublasSaxpy_v2(graph->GetDevice()->GetCuBLASHandle(), size,
				&alpha, dEdY_data, 1, dEdX_data, 1);
		}
		else
#endif
			cblas_saxpy(size, alpha, dEdY_data, 1, dEdX_data, 1);
	}

private:
	Shape shape_;
};

Expression Reshape(const Expression &x, const Shape &shape) {
	if (x.GetShape().GetSize() != shape.GetSize()) {
		REPORT_ERROR("Total size of input (%d) and requested shape (%d) mismatch.",
			x.GetShape().GetSize(), shape.GetSize());
	}
	Graph *graph = x.GetGraph();
	return graph->AddNode(new ReshapeNode(x.GetNodeIndex(), shape), shape);
}

Expression Flatten(const Expression &x) {
	return Reshape(x, Shape(x.GetShape().GetSize()));
}

class ReduceSumNodeCPU : public Node {
public:
	ReduceSumNodeCPU(int node, int axis) : Node{ node }, axis_(axis) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *x_data = (float*)x[0]->GetData();
		float *y_data = (float*)y->GetData();
		
		int size = x[0]->GetShape().GetSize();
		int batch_size = x[0]->GetBatchSize();
		if (axis_ == -1) {
			for (int batch_id = 0; batch_id < batch_size; batch_id++) {
				float sum = 0;
				for (int i = 0; i < size; i++)
					sum += x_data[batch_id * size + i];
				y_data[batch_id] = sum;
			}
		}
		else {
			int ndims = x[0]->GetShape().GetRank() + 1;
			GetTensorDims(x[0], x_dims_);
			GetTensorStrides(x[0], x_strides_);
			GetTensorStrides(y, y_strides_);
			auto transform_func = [&](int x_index, int y_index) {
				y_data[y_index] += x_data[x_index];
			};
			Transform<2>(transform_func, ndims, x_dims_, { x_strides_, y_strides_ });
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = (float*)dEdY->GetData();
		float *dEdX_data = (float*)dEdX[0]->GetData();
		int size = x[0]->GetShape().GetSize();
		int batch_size = x[0]->GetBatchSize();
		if (axis_ == -1) {
			for (int batch_id = 0; batch_id < batch_size; batch_id++)
				for (int i = 0; i < size; i++)
					dEdX_data[batch_id * size + i] += dEdY_data[batch_id];
		}
		else {
			int ndims = x[0]->GetShape().GetRank() + 1;
			auto transform_func = [&](int x_index, int y_index) {
				dEdX_data[x_index] += dEdY_data[y_index];
			};
			Transform<2>(transform_func, ndims, x_dims_, { x_strides_, y_strides_ });
		}
	}

private:
	int axis_;
	mutable int x_dims_[kMaxTensorDim + 1];
	mutable int x_strides_[kMaxTensorDim + 1], y_strides_[kMaxTensorDim + 1];
};

template<typename Dummy>
struct ReduceSumNodeFactory<Dummy, CPU> {
	Node *Create(int node, int axis) {
		return new ReduceSumNodeCPU(node, axis);
	}
};

Expression ReduceSum(const Expression &x, int axis) {
	Graph *graph = x.GetGraph();
	Shape output_shape;
	if (axis == -1)
		output_shape = Shape(1);
	else {
		output_shape = x.GetShape();
		output_shape.SetDim(axis, 1);
	}
	return CreateDeviceSpecificNode<ReduceSumNodeFactory>(graph, output_shape, x.GetNodeIndex(), axis);
}

class SliceNodeCPU : public Node {
public:
	SliceNodeCPU(int node, const Shape &start, const Shape &size) : Node{ node }, start_(start), size_(size) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		int ndims = x[0]->GetShape().GetRank() + 1;
		GetTensorStrides(x[0], x_strides_);
		GetTensorStrides(y, y_strides_);
		GetTensorDims(y, dims_);
		base_index_ = 0;
		for (int i = 1; i < ndims; i++)
			base_index_ += x_strides_[i] * start_.GetDim(i - 1);

		auto transform_func = [&](int x_index, int y_index) {
			y_data[y_index] = x_data[base_index_ + x_index];
		};
		Transform<2>(transform_func, ndims, dims_, { x_strides_, y_strides_ });
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int ndims = x[0]->GetShape().GetRank() + 1;

		auto transform_func = [&](int x_index, int y_index) {
			dEdX_data[base_index_ + x_index] += dEdY_data[y_index];
		};
		Transform<2>(transform_func, ndims, dims_, { x_strides_, y_strides_ });
	}

private:
	Shape start_, size_;
	mutable int x_strides_[kMaxTensorDim + 1], y_strides_[kMaxTensorDim + 1], dims_[kMaxTensorDim + 1];
	mutable int base_index_;
};

template<typename Dummy>
struct SliceNodeFactory<Dummy, CPU> {
	Node *Create(int node, const Shape &start, const Shape &size) {
		return new SliceNodeCPU(node, start, size);
	}
};

Expression Slice(const Expression &x, const Shape &start, const Shape &size) {
	const Shape &shape = x.GetShape();
	if (start.GetRank() != size.GetRank())
		REPORT_ERROR("Rank mismatch for start and size parameters.");
	if (shape.GetRank() != start.GetRank())
		REPORT_ERROR("Rank mismatch for input and given slicing range.");
	for (int i = 0; i < start.GetRank(); i++) {
		if (start.GetDim(i) < 0 || start.GetDim(i) >= shape.GetDim(i)
			|| start.GetDim(i) + size.GetDim(i) > shape.GetDim(i)) {
			REPORT_ERROR("Slicing out of range for dimension: %d. "
				"Input range: [0, %d). Requested range: [%d, %d).",
				i, 0, shape.GetDim(i), start.GetDim(i), start.GetDim(i) + size.GetDim(i));
		}
	}
	return CreateDeviceSpecificNode<SliceNodeFactory>(x.GetGraph(), size, x.GetNodeIndex(), start, size);
}

class ConcatNodeCPU : public Node {
public:
	ConcatNodeCPU(initializer_view<Expression> values, int axis) : Node(values), axis_(axis) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &y_shape = y->GetShape();
		float *y_data = y->GetData();
		int axis_stride = y_shape.GetSizeRange(axis_ + 1, y_shape.GetRank());
		int axis_total = y_shape.GetDim(axis_);
		int higher_stride = axis_stride * axis_total;
		int higher_size = y_shape.GetSizeRange(0, axis_);
		int base = 0;
		for (size_t i = 0; i < x.size(); i++) {
			int cur_axis_total = x[i]->GetShape().GetDim(axis_);
			int cur_axis_size = axis_stride * cur_axis_total;
			const float *x_data = x[i]->GetData();
			for (int h = 0; h < higher_size; h++)
				for (int l = 0; l < cur_axis_size; l++) {
					int index = base + h * higher_stride + l;
					y_data[index] = *x_data++;
				}
			base += cur_axis_total * axis_stride;
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const Shape &y_shape = y->GetShape();
		const float *dEdY_data = dEdY->GetData();
		int axis_stride = y_shape.GetSizeRange(axis_ + 1, y_shape.GetRank());
		int axis_total = y_shape.GetDim(axis_);
		int higher_stride = axis_stride * axis_total;
		int higher_size = y_shape.GetSizeRange(0, axis_);
		int base = 0;
		for (size_t i = 0; i < x.size(); i++) {
			int cur_axis_total = x[i]->GetShape().GetDim(axis_);
			int cur_axis_size = axis_stride * cur_axis_total;
			float *dEdX_data = dEdX[i]->GetData();
			for (int h = 0; h < higher_size; h++)
				for (int l = 0; l < cur_axis_size; l++) {
					int index = base + h * higher_stride + l;
					*dEdX_data++ += dEdY_data[index];
				}
			base += cur_axis_total * axis_stride;
		}
	}

private:
	int axis_;
};

template<typename Dummy>
struct ConcatNodeFactory<Dummy, CPU> {
	Node *Create(initializer_view<Expression> values, int axis) {
		return new ConcatNodeCPU(values, axis);
	}
};

Expression Concat(initializer_view<Expression> values, int axis) {
	if (values.size() == 0 || values.size() == 1)
		REPORT_ERROR("Must have at least two values for concatenation.");
	Graph *graph = values.begin()->GetGraph();
	Shape output_shape = values.begin()->GetShape();
	int sum = 0;
	for (const Expression &value : values) {
		if (value.GetBatchSize() != 1)
			REPORT_ERROR("Concat does not support batch input.");
		const Shape &shape = value.GetShape();
		if (shape.GetRank() != output_shape.GetRank())
			REPORT_ERROR("Concatenation tensor rank mismatch.");
		for (int i = 0; i < shape.GetRank(); i++)
			if (i != axis && shape.GetDim(i) != output_shape.GetDim(i))
				REPORT_ERROR("Dimension excluding concatenating axis mismatch.");
		sum += shape.GetDim(axis);
	}
	output_shape.SetDim(axis, sum);
	return CreateDeviceSpecificNode<ConcatNodeFactory>(graph, output_shape, values, axis);
}

class DropoutNodeCPU : public Node {
public:
	DropoutNodeCPU(int node, float p) : Node{ node }, p_(p) {}

	virtual void FreeMemory(Device *device) override {
		device->FreeMemory(scales_);
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		std::mt19937 gen(graph->GetIncrementRandomSeed());
		std::uniform_real_distribution<float> dist{ 0.f, 1.f };

		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		int size = x[0]->GetBatchSize() * x[0]->GetShape().GetSize();
		scales_ = (float *)graph->GetDevice()->AllocateMemory(size * sizeof(float));
		float scale = 1.f / (1.f - p_);
		for (int i = 0; i < size; i++) {
			if (dist(gen) <= p_)
				scales_[i] = 0.f;
			else
				scales_[i] = scale;
			y_data[i] = x_data[i] * scales_[i];
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int size = x[0]->GetBatchSize() * x[0]->GetShape().GetSize();
		for (int i = 0; i < size; i++)
			dEdX_data[i] += dEdY_data[i] / scales_[i];
	}

private:
	float p_;
	mutable float *scales_;
};

template<typename Dummy>
struct DropoutNodeFactory<Dummy, CPU> {
	Node *Create(int node, float p) {
		return new DropoutNodeCPU(node, p);
	}
};

Expression Dropout(const Expression &x, float p) {
	return CreateDeviceSpecificNode<DropoutNodeFactory>(x.GetGraph(), x.GetShape(), x.GetNodeIndex(), p);
}

class SoftmaxNodeCPU : public Node {
public:
	SoftmaxNodeCPU(int node) : Node{ node } {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		// y = exp(x_i) / sum(exp(x_i))
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetRank() - 1);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
		// Softmax function
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		for (int t = 0; t < size; t++) {
			// Calculate exp(x_i) and sum(exp(x_i))
			float sum = 0;
			float *cur_y = y_data;
			for (int i = 0; i < dim_size; i++) {
				float x_i = *x_data++;
				float e_x_i = exp(x_i);
				*cur_y++ = e_x_i;
				sum += e_x_i;
			}
			sum = 1.f / sum;
			// Normalize according to sum
			for (int i = 0; i < dim_size; i++) {
				float e_x_i = *y_data;
				*y_data++ = e_x_i * sum;
			}
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		// dY/dX_i = y_i*dEdy_i - y_i*sum_j{y_j*dEdy_j}
		const Shape &input_shape = x[0]->GetShape();
		int size = x[0]->GetShape().GetSizeRange(0, input_shape.GetRank() - 1);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int cur = 0;
		for (int t = 0; t < size; t++) {
			float sum = 0;
			for (int i = 0; i < dim_size; i++) {
				dEdX_data[cur + i] = dEdY_data[cur + i] * y_data[cur + i];
				sum += dEdX_data[cur + i];
			}
			for (int i = 0; i < dim_size; i++)
				dEdX_data[cur + i] -= y_data[cur + i] * sum;
			cur += dim_size;
		}
	}
};

template<typename Dummy>
struct SoftmaxNodeFactory<Dummy, CPU> {
	Node *Create(int node) {
		return new SoftmaxNodeCPU(node);
	}
};

Expression Softmax(const Expression &x) {
	Graph *graph = x.GetGraph();
	return CreateDeviceSpecificNode<SoftmaxNodeFactory>(graph, x.GetShape(), x.GetNodeIndex());
}

Expression SoftMargin(const Expression &x, const Expression &label) {
	return CreateBinaryOpNode<SoftMarginForward, SoftMarginBackward>(x, label);
}

Expression BinaryCrossEntropy(const Expression &x, const Expression &label) {
	return CreateBinaryOpNode<BinaryCrossEntropyForward, BinaryCrossEntropyBackward>(x, label);
}

Expression BinaryClassificationAccuracy(const Expression &x, const Expression &label) {
	return CreateBinaryOpNode<BinaryClassificationAccuracyForward, BinaryNoBackward>(x, label);
}

class CrossEntropyNodeCPU : public Node {
public:
	CrossEntropyNodeCPU(Graph *graph, int node, const std::vector<int> &labels) : Node{ node }, labels_(labels) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		// y = -log(x_k)
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetRank() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
		// Cross entropy loss
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		for (int label_index = 0; label_index < size; label_index++) {
			y_data[label_index] = -log(x_data[labels_[label_index]]);
			x_data += dim_size;
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		// dY/dX_k = -1/X_k
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetRank() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
		const float *x_data = x[0]->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int cur = 0;
		for (int label_index = 0; label_index < size; label_index++) {
			int label = labels_[label_index];
			dEdX_data[cur + label] += dEdY_data[label_index] * (-1.f / x_data[cur + label]);
			cur += dim_size;
		}
	}

private:
	std::vector<int> labels_;
};

template<typename Dummy>
struct CrossEntropyNodeFactory<Dummy, CPU> {
	Node *Create(Graph *graph, int node, const std::vector<int> &labels) {
		return new CrossEntropyNodeCPU(graph, node, labels);
	}
};

Expression CrossEntropy(const Expression &x, int size, const int *labels) {
	Shape shape = x.GetShape();
	shape.SetDim(shape.GetRank() - 1, 1);
	Graph *graph = x.GetGraph();
	std::vector<int> l;
	for (int i = 0; i < size; i++)
		l.push_back(labels[i]);
	return CreateDeviceSpecificNode<CrossEntropyNodeFactory>(graph, shape, graph, x.GetNodeIndex(), l);
}

class ClassificationAccuracyNodeCPU : public Node {
public:
	ClassificationAccuracyNodeCPU(Graph *graph, int node, const std::vector<int> &labels) : Node{ node }, labels_(labels) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		// y = -log(x_k)
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetRank() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
		// Cross entropy loss
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		for (int label_index = 0; label_index < size; label_index++) {
			int max_index = 0;
			for (int i = 1; i < dim_size; i++) {
				if (x_data[i] > x_data[max_index])
					max_index = i;
			}
			if (max_index == labels_[label_index])
				y_data[label_index] = 1.f;
			else
				y_data[label_index] = 0.f;
			x_data += dim_size;
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		REPORT_ERROR("Backward propagation is unsupported for this expression.");
	}

private:
	std::vector<int> labels_;
};

template<typename Dummy>
struct ClassificationAccuracyNodeFactory<Dummy, CPU> {
	Node *Create(Graph *graph, int node, const std::vector<int> &labels) {
		return new ClassificationAccuracyNodeCPU(graph, node, labels);
	}
};

Expression ClassificationAccuracy(const Expression &x, int size, const int *labels) {
	Shape shape = x.GetShape();
	shape.SetDim(shape.GetRank() - 1, 1);
	Graph *graph = x.GetGraph();
	std::vector<int> l;
	for (int i = 0; i < size; i++)
		l.push_back(labels[i]);
	return CreateDeviceSpecificNode<ClassificationAccuracyNodeFactory>(graph, shape, graph, x.GetNodeIndex(), l);
}

#include "Device.h"
#include "Expression.h"
#include "Expression_p.h"
#include "Graph.h"
#include "Node.h"
#include "Utils.h"

#include <cblas.h>
#include <cstdlib>
#include <cstring>
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

static void *PinMemory(Device *device, const void *data, int size) {
	void *ret;
#ifdef USE_CUDA
	if (device->GetDeviceType() == GPU)
		ret = (float*)device->AllocateMemory(size, Device::PinnedScratchMemoryPool);
	else
#endif
		ret = (float*)device->AllocateMemory(size, Device::ScratchMemoryPool);
	memcpy(ret, data, size);
	return ret;
}

static void CopyMemoryHostToDeviceAsync(Device *device, void *dst, const void *src, int size) {
#ifdef USE_CUDA
	if (device->GetDeviceType() == GPU)
		CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice));
	else
#endif
		memcpy(dst, src, size);
}

class InputNode : public Node {
public:
	InputNode(Graph *graph, int batch_size, const Shape &shape, const float *data)
		: Node{}, batch_size_(batch_size), shape_(shape) {
		int size = batch_size * shape.GetSize() * sizeof(float);
		data_ = (float *)PinMemory(graph->GetDevice(), data, size);
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
		if (shape.GetDimCount() != 1)
			REPORT_ERROR("Shape of sparse input must be 1D.");
		sparse_data_ = (float *)PinMemory(graph->GetDevice(), sparse_data, nonzero_count * sizeof(float));
		batch_indices_ = (int *)PinMemory(graph->GetDevice(), batch_indices, (batch_size + 1) * sizeof(int));
		indices_ = (int *)PinMemory(graph->GetDevice(), indices, nonzero_count * sizeof(int));
	}

	virtual int GetFlags() const override {
		return NoAllocateForwardOutput;
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		Device *device = graph->GetDevice();
		float *sparse_data = (float*)device->AllocateMemory(nonzero_count_ * sizeof(float), Device::ScratchMemoryPool);
		int *batch_indices = (int*)device->AllocateMemory((batch_size_ + 1) * sizeof(int), Device::ScratchMemoryPool);
		int *indices = (int*)device->AllocateMemory(nonzero_count_ * sizeof(int), Device::ScratchMemoryPool);
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

template<typename ForwardFunc, typename BackwardFunc>
class BinaryOpNodeCPU : public Node {
public:
	BinaryOpNodeCPU(int lhs_node, int rhs_node) : Node{ lhs_node, rhs_node } {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		int size = y->GetShape().GetSize();
		float *y_data = y->GetData();
		int left_batch_size = x[0]->GetBatchSize(), right_batch_size = x[1]->GetBatchSize();
		// TODO: Simplify broadcast logic, avoid repeating code
		if (left_batch_size == right_batch_size) {
			size *= left_batch_size;
			for (int i = 0; i < size; i++)
				y_data[i] = ForwardFunc()(lhs_data[i], rhs_data[i]);
		}
		else if (left_batch_size == 1) {
			// Broadcast left
			int j = 0;
			for (int batch_id = 0; batch_id < right_batch_size; batch_id++)
				for (int i = 0; i < size; i++) {
					y_data[j] = ForwardFunc()(lhs_data[i], rhs_data[j]);
					j++;
				}
		}
		else if (right_batch_size == 1) {
			// Broadcast right
			int j = 0;
			for (int batch_id = 0; batch_id < left_batch_size; batch_id++)
				for (int i = 0; i < size; i++) {
					y_data[j] = ForwardFunc()(lhs_data[j], rhs_data[i]);
					j++;
				}
		}
		else
			DEBUG_BREAK();
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value)
			REPORT_ERROR("Backward propagation is unsupported for this expression.");

		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdL_data = dEdX[0]->GetData(), *dEdR_data = dEdX[1]->GetData();
		int size = y->GetShape().GetSize();
		int left_batch_size = x[0]->GetBatchSize(), right_batch_size = x[1]->GetBatchSize();
		if (left_batch_size == right_batch_size) {
			size *= left_batch_size;
			for (int i = 0; i < size; i++) {
				float dYdL, dYdR;
				BackwardFunc()(lhs_data[i], rhs_data[i], y_data[i], &dYdL, &dYdR);
				dEdL_data[i] += dYdL * dEdY_data[i];
				dEdR_data[i] += dYdR * dEdY_data[i];
			}
		}
		else if (left_batch_size == 1) {
			// Broadcast left
			int j = 0;
			for (int batch_id = 0; batch_id < right_batch_size; batch_id++)
				for (int i = 0; i < size; i++) {
					float dYdL, dYdR;
					BackwardFunc()(lhs_data[i], rhs_data[j], y_data[j], &dYdL, &dYdR);
					dEdL_data[i] += dYdL * dEdY_data[j];
					dEdR_data[j] += dYdR * dEdY_data[j];
					j++;
				}
		}
		else if (right_batch_size == 1) {
			// Broadcast right
			int j = 0;
			for (int batch_id = 0; batch_id < left_batch_size; batch_id++)
				for (int i = 0; i < size; i++) {
					float dYdL, dYdR;
					BackwardFunc()(lhs_data[j], rhs_data[i], y_data[j], &dYdL, &dYdR);
					dEdL_data[j] += dYdL * dEdY_data[j];
					dEdR_data[i] += dYdR * dEdY_data[j];
					j++;
				}
		}
		else
			DEBUG_BREAK();
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
	if (lhs_shape.GetDimCount() != rhs_shape.GetDimCount()) {
		REPORT_ERROR("Input operands have different ranks (%d and %d).",
			lhs_shape.GetDimCount(), rhs_shape.GetDimCount());
	}
	int ndims = lhs_shape.GetDimCount();
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

Expression operator-(const Expression &lhs, const Expression &rhs) {
	return CreateBinaryOpNode<ElemSubForward, ElemSubBackward>(lhs, rhs);
}

Expression operator*(const Expression &lhs, const Expression &rhs) {
	return CreateBinaryOpNode<ElemMulForward, ElemMulBackward>(lhs, rhs);
}

Expression operator/(const Expression &lhs, const Expression &rhs) {
	return CreateBinaryOpNode<ElemDivForward, ElemDivBackward>(lhs, rhs);
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

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		// y = L * R
		const Shape &lhs_shape = x[0]->GetShape();
		const Shape &rhs_shape = x[1]->GetShape();
		int M = lhs_shape.GetDimCount() == 2 ? lhs_shape.GetDim(0) : 1;
		int K = rhs_shape.GetDim(0);
		int N = rhs_shape.GetDimCount() == 2 ? rhs_shape.GetDim(1) : 1;
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		float *y_data = y->GetData();
		if (x[0]->GetBatchSize() == 1 && rhs_shape.GetDimCount() == 1) {
			SGEMM(graph->GetDevice(), CblasNoTrans, CblasTrans, CblasTrans,
				M, N * x[1]->GetBatchSize(), K, 1.f, lhs_data, rhs_data, 0.f, y_data);
		}
		else {
			BatchSGEMM(graph->GetDevice(), CblasNoTrans, CblasNoTrans,
				M, N, K,
				1.f, lhs_data, rhs_data, 0.f, y_data,
				x[0]->GetBatchSize(), x[1]->GetBatchSize(), y->GetBatchSize());
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const Shape &lhs_shape = x[0]->GetShape();
		const Shape &rhs_shape = x[1]->GetShape();
		int M = lhs_shape.GetDimCount() == 2 ? lhs_shape.GetDim(0) : 1;
		int K = rhs_shape.GetDim(0);
		int N = rhs_shape.GetDimCount() == 2 ? rhs_shape.GetDim(1) : 1;
		const float *dEdY_data = dEdY->GetData();
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		float *dEdL_data = dEdX[0]->GetData(), *dEdR_data = dEdX[1]->GetData();
		// dEdL += dEdY * R'
		// dEdR += L' * dEdY
		if (x[0]->GetBatchSize() == 1 && rhs_shape.GetDimCount() == 1) {
			SGEMM(graph->GetDevice(), CblasTrans, CblasNoTrans, CblasNoTrans,
				M, K, N * x[1]->GetBatchSize(),
				1.f, dEdY_data, rhs_data, 1.f, dEdL_data);
			SGEMM(graph->GetDevice(), CblasTrans, CblasTrans, CblasTrans,
				K, N * x[1]->GetBatchSize(), M,
				1.f, lhs_data, dEdY_data, 1.f, dEdR_data);
		}
		else {
			BatchSGEMM(graph->GetDevice(), CblasNoTrans, CblasTrans,
				M, K, N,
				1.f, dEdY_data, rhs_data, 1.f, dEdL_data,
				dEdY->GetBatchSize(), x[1]->GetBatchSize(), dEdX[0]->GetBatchSize());
			BatchSGEMM(graph->GetDevice(), CblasTrans, CblasNoTrans,
				K, N, M,
				1.f, lhs_data, dEdY_data, 1.f, dEdR_data,
				x[0]->GetBatchSize(), dEdY->GetBatchSize(), dEdX[1]->GetBatchSize());
		}
	}
};

Expression MatMul(const Expression &lhs, const Expression &rhs) {
	/* Infer output shape. */
	const Shape &lhs_shape = lhs.GetShape(), &rhs_shape = rhs.GetShape();
	if (lhs_shape.GetDimCount() > 2)
		REPORT_ERROR("Left operand is not a vector or matrix.");
	if (rhs_shape.GetDimCount() > 2)
		REPORT_ERROR("Right operand is not a vector or matrix.");
	if (lhs_shape.GetDimCount() == 1 && rhs_shape.GetDimCount() == 1)
		REPORT_ERROR("Left and right operands are both vectors, use Dot() for now.");
	Shape output_shape;
	if (lhs_shape.GetDimCount() == 1) {
		if (lhs_shape.GetDim(0) != rhs_shape.GetDim(0)) {
			REPORT_ERROR("Dimension mismatch for vector-matrix multiplication: (%d) * (%d, %d).",
				lhs_shape.GetDim(0), rhs_shape.GetDim(0), rhs_shape.GetDim(1));
		}
		output_shape = Shape(rhs_shape.GetDim(1));
	}
	else if (rhs_shape.GetDimCount() == 1) {
		if (lhs_shape.GetDim(1) != rhs_shape.GetDim(0)) {
			REPORT_ERROR("Dimension mismatch for matrix-vector multiplication: (%d) * (%d, %d).",
				lhs_shape.GetDim(0), lhs_shape.GetDim(1), rhs_shape.GetDim(0));
		}
		output_shape = Shape(lhs_shape.GetDim(0));
	}
	else {
		if (lhs_shape.GetDim(1) != rhs_shape.GetDim(0)) {
			REPORT_ERROR("Dimension mismatch for matrix-matrix multiplication: (%d, %d) * (%d, %d).",
				lhs_shape.GetDim(0), lhs_shape.GetDim(1), rhs_shape.GetDim(0), rhs_shape.GetDim(1));
		}
		output_shape = Shape(lhs_shape.GetDim(0), rhs_shape.GetDim(1));
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
	if (lhs_shape.GetDimCount() != 1 || rhs_shape.GetDimCount() != 1)
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

	if (x_shape.GetDimCount() < 2)
		REPORT_ERROR("Input must have at least rank 2.");
	int dims = x_shape.GetDimCount() - 1;
	if (filter_shape.GetDimCount() != filter_window_offset + dims)
		REPORT_ERROR("Incompatible filter shape.");
	if (strides.GetDimCount() != dims)
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
		int dims = y->GetShape().GetDimCount() - 1;

		if (graph->GetDeviceType() == CPU)
			REPORT_ERROR("Convolution is only implemented in GPU.");

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

class PoolingNode : public Node {
public:
	enum PoolingMode {
		MaxPooling,
		AvgPooling,
		AvgPoolingWithPadding,
	};

	PoolingNode(int node, const Shape &filter_shape, const Shape &strides, const Shape &padding, PoolingMode mode):
		Node{ node }, filter_shape_(filter_shape), strides_(strides), padding_(padding), mode_(mode) {
		CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc_));
		CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_));
		CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_));
	}

	virtual ~PoolingNode() {
		CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_desc_));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc_));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc_));
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &x_shape = x[0]->GetShape();
		const Shape &y_shape = y->GetShape();
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		int ndims = y->GetShape().GetDimCount() - 1;

		if (graph->GetDeviceType() == CPU)
			REPORT_ERROR("Pooling is only implemented in GPU.");

		int x_dims[CUDNN_DIM_MAX], y_dims[CUDNN_DIM_MAX];
		x_dims[0] = x[0]->GetBatchSize();
		y_dims[0] = y->GetBatchSize();
		for (int i = 0; i < ndims + 1; i++) {
			x_dims[i + 1] = x_shape.GetDim(i);
			y_dims[i + 1] = y_shape.GetDim(i);
		}
		int x_strides[CUDNN_DIM_MAX], y_strides[CUDNN_DIM_MAX];
		x_strides[ndims + 1] = 1;
		y_strides[ndims + 1] = 1;
		for (int i = ndims; i >= 0; i--) {
			x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
			y_strides[i] = y_dims[i + 1] * y_strides[i + 1];
		}
		cudnnPoolingMode_t pooling_mode;
		if (mode_ == MaxPooling)
			pooling_mode = CUDNN_POOLING_MAX;
		else if (mode_ == AvgPooling)
			pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
		else if (mode_ == AvgPoolingWithPadding)
			pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
		else
			DEBUG_BREAK();
		CUDNN_CALL(cudnnSetPoolingNdDescriptor(pooling_desc_, pooling_mode, CUDNN_PROPAGATE_NAN,
			ndims, filter_shape_.data(), padding_.data(), strides_.data()));
		CUDNN_CALL(cudnnSetTensorNdDescriptor(x_desc_, CUDNN_DATA_FLOAT, ndims + 2, x_dims, x_strides));
		CUDNN_CALL(cudnnSetTensorNdDescriptor(y_desc_, CUDNN_DATA_FLOAT, ndims + 2, y_dims, y_strides));
		float alpha = 1.f, beta = 0.f;
		CUDNN_CALL(cudnnPoolingForward(graph->GetDevice()->GetCuDNNHandle(), pooling_desc_,
			&alpha, x_desc_, x_data, &beta, y_desc_, y_data));
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *x_data = x[0]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();

		if (graph->GetDeviceType() == CPU)
			REPORT_ERROR("Pooling is only implemented in GPU.");
		
		float alpha = 1.f, beta = 1.f;
		CUDNN_CALL(cudnnPoolingBackward(graph->GetDevice()->GetCuDNNHandle(), pooling_desc_,
			&alpha, y_desc_, y_data, y_desc_, dEdY_data, x_desc_, x_data, &beta,
			x_desc_, dEdX_data));
	}

private:
	Shape filter_shape_, strides_, padding_;
	PoolingMode mode_;
#ifdef USE_CUDA
	cudnnPoolingDescriptor_t pooling_desc_;
	cudnnTensorDescriptor_t x_desc_, y_desc_;
#endif
};

Expression MaxPooling(const Expression &x, const Shape &filter_shape, const Shape &strides, const Shape &padding) {
	Shape output_shape = FilterForwardShape(x.GetShape(), filter_shape, strides, padding, true);
	Graph *graph = x.GetGraph();
	return graph->AddNode(new PoolingNode(x.GetNodeIndex(), filter_shape, strides, padding, PoolingNode::MaxPooling), output_shape);
}

Expression AvgPooling(const Expression &x, const Shape &filter_shape, const Shape &strides, const Shape &padding) {
	Shape output_shape = FilterForwardShape(x.GetShape(), filter_shape, strides, padding, true);
	Graph *graph = x.GetGraph();
	return graph->AddNode(new PoolingNode(x.GetNodeIndex(), filter_shape, strides, padding, PoolingNode::AvgPooling), output_shape);
}

class ReshapeNode : public Node {
public:
	ReshapeNode(int node, const Shape &shape) : Node{ node }, shape_(shape) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		float *x_data = (float*)x[0]->GetData();
		*y = Tensor(graph->GetDeviceType(), x[0]->GetBatchSize(), shape_, x_data);
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
#endif
		else
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
	ReduceSumNodeCPU(int node) : Node{ node } {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *x_data = (float*)x[0]->GetData();
		float *y_data = (float*)y->GetData();
		
		int size = x[0]->GetShape().GetSize();
		int batch_size = x[0]->GetBatchSize();
		for (int batch_id = 0; batch_id < batch_size; batch_id++) {
			float sum = 0;
			for (int i = 0; i < size; i++)
				sum += x_data[batch_id * size + i];
			y_data[batch_id] = sum;
		}
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = (float*)dEdY->GetData();
		float *dEdX_data = (float*)dEdX[0]->GetData();
		int size = x[0]->GetShape().GetSize();
		int batch_size = x[0]->GetBatchSize();
		for (int batch_id = 0; batch_id < batch_size; batch_id++)
			for (int i = 0; i < size; i++)
				dEdX_data[batch_id * size + i] += dEdY_data[batch_id];
	}
};

template<typename Dummy>
struct ReduceSumNodeFactory<Dummy, CPU> {
	Node *Create(int node) {
		return new ReduceSumNodeCPU(node);
	}
};

Expression ReduceSum(const Expression &x) {
	Graph *graph = x.GetGraph();
	return CreateDeviceSpecificNode<ReduceSumNodeFactory>(graph, Shape(1), x.GetNodeIndex());
}

class SliceNodeCPU : public Node {
public:
	SliceNodeCPU(int node, const Shape &start, const Shape &size) : Node{ node }, start_(start), size_(size) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		if (x[0]->GetBatchSize() != 1)
			DEBUG_BREAK();
		float *ptr = x[0]->GetData() + start_.GetDim(0);
		int size = size_.GetDim(0) * sizeof(float);
		graph->GetDevice()->CopyMemory(y->GetData(), ptr, size);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int count = size_.GetDim(0);
		float *ptr = dEdX_data + start_.GetDim(0);
		cblas_saxpy(count, 1.f, dEdY_data, 1, ptr, 1);
	}

private:
	Shape start_, size_;
};

template<typename Dummy>
struct SliceNodeFactory<Dummy, CPU> {
	Node *Create(int node, const Shape &start, const Shape &size) {
		return new SliceNodeCPU(node, start, size);
	}
};

Expression Slice(const Expression &x, const Shape &start, const Shape &size) {
	const Shape &shape = x.GetShape();
	if (start.GetDimCount() != size.GetDimCount())
		REPORT_ERROR("Rank mismatch for start and size parameters.");
	if (shape.GetDimCount() != start.GetDimCount())
		REPORT_ERROR("Rank mismatch for input and given slicing range.");
	for (int i = 0; i < start.GetDimCount(); i++) {
		if (start.GetDim(i) < 0 || start.GetDim(i) >= shape.GetDim(i)
			|| start.GetDim(i) + size.GetDim(i) > shape.GetDim(i)) {
			REPORT_ERROR("Slicing out of range for dimension: %d. "
				"Input range: [0, %d). Requested range: [%d, %d).",
				i, 0, shape.GetDim(i), start.GetDim(i), start.GetDim(i) + size.GetDim(i));
		}
	}
	return CreateDeviceSpecificNode<SliceNodeFactory>(x.GetGraph(), size, x.GetNodeIndex(), start, size);
}

class SoftmaxNodeCPU : public Node {
public:
	SoftmaxNodeCPU(int node) : Node{ node } {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		// y = exp(x_i) / sum(exp(x_i))
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetDimCount() - 1);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
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
		int size = x[0]->GetShape().GetSizeRange(0, input_shape.GetDimCount() - 1);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
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
		int size = input_shape.GetSizeRange(0, input_shape.GetDimCount() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
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
		int size = input_shape.GetSizeRange(0, input_shape.GetDimCount() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
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
	shape.SetDim(shape.GetDimCount() - 1, 1);
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
		int size = input_shape.GetSizeRange(0, input_shape.GetDimCount() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
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
	shape.SetDim(shape.GetDimCount() - 1, 1);
	Graph *graph = x.GetGraph();
	std::vector<int> l;
	for (int i = 0; i < size; i++)
		l.push_back(labels[i]);
	return CreateDeviceSpecificNode<ClassificationAccuracyNodeFactory>(graph, shape, graph, x.GetNodeIndex(), l);
}

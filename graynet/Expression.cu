#include "Device.h"
#include "Graph.h"
#include "Expression.h"
#include "Expression_p.h"
#include "Node.h"
#include "Utils.h"

#include <cudnn.h>

static const int kThreadsPerBlock = 32;

static inline __device__ int GetTensorStorageIndex(int logical_index, int ndims, int *elems, int *strides) {
	int ret = 0;
	for (int i = 0; i < ndims; i++) {
		int cur = logical_index / elems[i];
		ret += strides[i] * cur;
		logical_index %= elems[i];
	}
	return ret;
}

struct BinaryForwardDims {
	int elems[kMaxTensorDim + 1];
	int lhs_strides[kMaxTensorDim + 1], rhs_strides[kMaxTensorDim + 1];
};

template<typename ForwardFunc>
static __global__ void BinaryForwardKernel(const float *lhs, const float *rhs, float *y,
	int nelems, int ndims, BinaryForwardDims forward) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nelems) {
		int lhs_index = GetTensorStorageIndex(i, ndims, forward.elems, forward.lhs_strides);
		int rhs_index = GetTensorStorageIndex(i, ndims, forward.elems, forward.rhs_strides);
		y[i] = ForwardFunc()(lhs[lhs_index], rhs[rhs_index]);
	}
}

struct BinaryBackwardDims {
	int elems[kMaxTensorDim + 1];
	int const_elems[kMaxTensorDim + 1], reduce_elems[kMaxTensorDim + 1];
	int lhs_strides[kMaxTensorDim + 1], rhs_strides[kMaxTensorDim + 1];
};

template<typename BackwardFunc>
static __global__ void BinaryBackwardKernel(const float *lhs, const float *rhs, const float *y,
	const float *dEdY, float *dEdL, float *dEdR, int nconst_elems, int nreduce_elems, int ndims,
	BinaryBackwardDims backward) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nconst_elems) {
		int base_index = GetTensorStorageIndex(i, ndims, backward.const_elems, backward.elems);
		for (int j = 0; j < nreduce_elems; j++) {
			int index = base_index + GetTensorStorageIndex(j, ndims, backward.reduce_elems, backward.elems);
			int lhs_index = GetTensorStorageIndex(index, ndims, backward.elems, backward.lhs_strides);
			int rhs_index = GetTensorStorageIndex(index, ndims, backward.elems, backward.rhs_strides);
			float dYdL_value, dYdR_value;
			BackwardFunc()(lhs[lhs_index], rhs[rhs_index], y[index], &dYdL_value, &dYdR_value);
			float dEdY_value = dEdY[index];
			dEdL[lhs_index] += dEdY_value * dYdL_value;
			dEdR[rhs_index] += dEdY_value * dYdR_value;
		}
	}
}

static void GetTensorStrides(const Tensor *tensor, int strides[kMaxTensorDim + 1]) {
	int batch_size = tensor->GetBatchSize();
	const Shape &shape = tensor->GetShape();
	int ndims = 1 + shape.GetDimCount();
	int cur = 1;
	for (int i = ndims - 1; i >= 1; i--) {
		if (shape.GetDim(i - 1) == 1)
			strides[i] = 0;
		else {
			strides[i] = cur;
			cur *= shape.GetDim(i - 1);
		}
	}
	strides[0] = (batch_size == 1) ? 0 : cur;
}

template<typename ForwardFunc, typename BackwardFunc>
class BinaryOpNode<GPU, ForwardFunc, BackwardFunc> : public Node {
public:
	BinaryOpNode(int lhs_node, int rhs_node) : Node{ lhs_node, rhs_node } {}

	virtual Shape ForwardShape(const std::vector<Shape> &x_shapes) const override {
		const Shape &lhs_shape = x_shapes[0];
		const Shape &rhs_shape = x_shapes[1];
		// Broadcasting
		if (lhs_shape.GetDimCount() != rhs_shape.GetDimCount())
			abort();
		int ndims = lhs_shape.GetDimCount();
		Shape shape;
		for (int i = 0; i < ndims; i++) {
			if (lhs_shape.GetDim(i) == 1)
				shape.PushDim(rhs_shape.GetDim(i));
			else if (rhs_shape.GetDim(i) == 1)
				shape.PushDim(lhs_shape.GetDim(i));
			else if (lhs_shape.GetDim(i) == rhs_shape.GetDim(i))
				shape.PushDim(lhs_shape.GetDim(i));
			else
				abort();
		}
		return shape;
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		int size = y->GetShape().GetSize();
		float *y_data = y->GetData();
		int y_batch_size = y->GetBatchSize();
		const Shape &y_shape = y->GetShape();

		int nelems = y_batch_size * y_shape.GetSize();
		int ndims = 1 + y_shape.GetDimCount();
		BinaryForwardDims forward;
		forward.elems[ndims - 1] = 1;
		for (int i = ndims - 2; i >= 0; i--)
			forward.elems[i] = forward.elems[i + 1] * y_shape.GetDim(i);
		GetTensorStrides(x[0], forward.lhs_strides);
		GetTensorStrides(x[1], forward.rhs_strides);
		
		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (nelems + threadsPerBlock - 1) / threadsPerBlock;
		BinaryForwardKernel<ForwardFunc><<<blocksPerGrid, threadsPerBlock>>>(
			lhs_data, rhs_data, y_data, nelems, ndims, forward);
		CUDA_CALL(cudaGetLastError());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdL_data = dEdX[0]->GetData(), *dEdR_data = dEdX[1]->GetData();
		const Shape &lhs_shape = x[0]->GetShape(), &rhs_shape = x[1]->GetShape();
		const Shape &y_shape = y->GetShape();
		int ndims = 1 + y_shape.GetDimCount();

		BinaryBackwardDims backward;
		// Fill elems[]
		backward.elems[ndims - 1] = 1;
		for (int i = ndims - 2; i >= 0; i--)
			backward.elems[i] = backward.elems[i + 1] * y_shape.GetDim(i);
		// Fill const_elems[] and reduce_elems[]
		int nconst_elems = 1, nreduce_elems = 1;
		for (int i = ndims - 1; i >= 1; i--) {
			backward.const_elems[i] = nconst_elems;
			backward.reduce_elems[i] = nreduce_elems;
			// Const dimension or reduce dimension ?
			if (lhs_shape.GetDim(i - 1) == rhs_shape.GetDim(i - 1))
				nconst_elems *= y_shape.GetDim(i - 1);
			else
				nreduce_elems *= y_shape.GetDim(i - 1);
		}
		backward.const_elems[0] = nconst_elems;
		backward.reduce_elems[0] = nreduce_elems;
		if (x[0]->GetBatchSize() == x[1]->GetBatchSize())
			nconst_elems *= y->GetBatchSize();
		else
			nreduce_elems *= y->GetBatchSize();
		// Fill lhs_strides[] and rhs_strides[]
		GetTensorStrides(x[0], backward.lhs_strides);
		GetTensorStrides(x[1], backward.rhs_strides);

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (nconst_elems + kThreadsPerBlock - 1) / kThreadsPerBlock;
		BinaryBackwardKernel<BackwardFunc><<<blocksPerGrid, threadsPerBlock>>>(
			lhs_data, rhs_data, y_data, dEdY_data, dEdL_data, dEdR_data,
			nconst_elems, nreduce_elems, ndims, backward);
		CUDA_CALL(cudaGetLastError());
	}
};

template<typename ForwardFunc>
static __global__ void BinaryLeftScalarForwardKernel(float lhs, const float *rhs, float *y, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		y[i] = ForwardFunc()(lhs, rhs[i]);
}

template<typename BackwardFunc>
static __global__ void BinaryLeftScalarBackwardKernel(float lhs, const float *rhs, const float *y,
	const float *dEdY, float *dEdR, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		float dYdL, dYdR;
		BackwardFunc()(lhs, rhs[i], y[i], &dYdL, &dYdR);
		dEdR[i] = dEdY[i] * dYdR;
	}
}

template<typename ForwardFunc, typename BackwardFunc>
class BinaryLeftScalarOpNode<GPU, ForwardFunc, BackwardFunc> : public Node {
public:
	BinaryLeftScalarOpNode(float lhs_scalar, int rhs_node) : Node{ rhs_node }, lhs_scalar_(lhs_scalar) {}

	virtual Shape ForwardShape(const std::vector<Shape> &x_shapes) const override {
		return x_shapes[0];
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *rhs_data = x[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		float *y_data = y->GetData();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + kThreadsPerBlock - 1) / kThreadsPerBlock;
		BinaryLeftScalarForwardKernel<ForwardFunc><<<blocksPerGrid, threadsPerBlock>>>(
			lhs_scalar_, rhs_data, y_data, size);
		CUDA_CALL(cudaGetLastError());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *rhs_data = x[0]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdR_data = dEdX[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + kThreadsPerBlock - 1) / kThreadsPerBlock;
		BinaryLeftScalarBackwardKernel<BackwardFunc><<<blocksPerGrid, threadsPerBlock>>>(
			lhs_scalar_, rhs_data, y_data, dEdY_data, dEdR_data, size);
		CUDA_CALL(cudaGetLastError());
	}

private:
	float lhs_scalar_;
};

template<typename ForwardFunc>
static __global__ void BinaryRightScalarForwardKernel(const float *lhs, float rhs, float *y, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		y[i] = ForwardFunc()(lhs[i], rhs);
}

template<typename BackwardFunc>
static __global__ void BinaryRightScalarBackwardKernel(const float *lhs, float rhs, const float *y,
	const float *dEdY, float *dEdL, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		float dYdL, dYdR;
		BackwardFunc()(lhs[i], rhs, y[i], &dYdL, &dYdR);
		dEdL[i] = dEdY[i] * dYdL;
	}
}

template<typename ForwardFunc, typename BackwardFunc>
class BinaryRightScalarOpNode<GPU, ForwardFunc, BackwardFunc> : public Node {
public:
	BinaryRightScalarOpNode(int lhs_node, float rhs_scalar) : Node{ lhs_node }, rhs_scalar_(rhs_scalar) {}

	virtual Shape ForwardShape(const std::vector<Shape> &x_shapes) const override {
		return x_shapes[0];
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *lhs_data = x[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		float *y_data = y->GetData();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + kThreadsPerBlock - 1) / kThreadsPerBlock;
		BinaryRightScalarForwardKernel<ForwardFunc><<<blocksPerGrid, threadsPerBlock>>>(
			lhs_data, rhs_scalar_, y_data, size);
		CUDA_CALL(cudaGetLastError());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *lhs_data = x[0]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdL_data = dEdX[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + kThreadsPerBlock - 1) / kThreadsPerBlock;
		BinaryRightScalarBackwardKernel<BackwardFunc><<<blocksPerGrid, threadsPerBlock>>>(
			lhs_data, rhs_scalar_, y_data, dEdY_data, dEdL_data, size);
		CUDA_CALL(cudaGetLastError());
	}

private:
	float rhs_scalar_;
};

template<typename ForwardFunc>
static __global__ void UnaryForwardKernel(const float *x, float *y, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		y[i] = ForwardFunc()(x[i]);
}

template<typename BackwardFunc>
static __global__ void UnaryBackwardKernel(const float *x, const float *y,
	const float *dEdY, float *dEdX, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		float dYdX;
		BackwardFunc()(x[i], y[i], &dYdX);
		dEdX[i] = dEdY[i] * dYdX;
	}
}

template<typename ForwardFunc, typename BackwardFunc>
class UnaryOpNode<GPU, ForwardFunc, BackwardFunc> : public Node {
public:
	UnaryOpNode(int node) : Node{ node } {}

	virtual Shape ForwardShape(const std::vector<Shape> &x_shapes) const override {
		return x_shapes[0];
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *x_data = x[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();
		float *y_data = y->GetData();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + kThreadsPerBlock - 1) / kThreadsPerBlock;
		UnaryForwardKernel<ForwardFunc><<<blocksPerGrid, threadsPerBlock>>>(x_data, y_data, size);
		CUDA_CALL(cudaGetLastError());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *x_data = x[0]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int size = y->GetShape().GetSize() * x[0]->GetBatchSize();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + kThreadsPerBlock - 1) / kThreadsPerBlock;
		UnaryBackwardKernel<BackwardFunc><<<blocksPerGrid, threadsPerBlock>>>(
			x_data, y_data, dEdY_data, dEdX_data, size);
		CUDA_CALL(cudaGetLastError());
	}
};

INSTANTIATE_BINARY_OPS(GPU)
INSTANTIATE_BINARY_LEFT_SCALAR_OPS(GPU)
INSTANTIATE_BINARY_RIGHT_SCALAR_OPS(GPU)
INSTANTIATE_UNARY_OPS(GPU)

template<typename Dummy>
class SoftmaxNode<Dummy, GPU> : public Node {
public:
	SoftmaxNode(int node) : Node{ node } {}

	virtual Shape ForwardShape(const std::vector<Shape> &x_shapes) const override {
		return x_shapes[0];
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		// y = exp(x_i) / sum(exp(x_i))
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetDimCount() - 1);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();

		float alpha = 1.f, beta = 0.f;
		cudnnHandle_t cudnn_handle = graph->GetDevice()->GetCuDNNHandle();
		cudnnTensorDescriptor_t tensor_desc;
		CUDNN_CALL(cudnnCreateTensorDescriptor(&tensor_desc));
		CUDNN_CALL(cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, size, dim_size, 1, 1));
		CUDNN_CALL(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha, tensor_desc, x_data, &beta, tensor_desc, y_data));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(tensor_desc));
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

		float alpha = 1.f, beta = 1.f;
		cudnnHandle_t cudnn_handle = graph->GetDevice()->GetCuDNNHandle();
		cudnnTensorDescriptor_t tensor_desc;
		CUDNN_CALL(cudnnCreateTensorDescriptor(&tensor_desc));
		CUDNN_CALL(cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, size, dim_size, 1, 1));
		CUDNN_CALL(cudnnSoftmaxBackward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha, tensor_desc, y_data, tensor_desc, dEdY_data, &beta, tensor_desc, dEdX_data));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(tensor_desc));
	}
};

template class SoftmaxNode<void, GPU>;

static __global__ void CrossEntropyForward(const float *x, float *y, const int *labels, int N, int dim_size) {
	// y = -log(x_k)
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		int label = labels[i];
		y[i] = -log(x[dim_size * i + label]);
	}
}

static __global__ void CrossEntropyBackward(const float *x, const int *labels,
	const float *dEdY, float *dEdX, int N, int dim_size) {
	// dY/dX_k = -1/X_k
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		int label = labels[i];
		dEdX[dim_size * i + label] -= dEdY[i] * (1.f / x[dim_size * i + label]);
	}
}

template<typename Dummy>
class CrossEntropyNode<Dummy, GPU> : public Node {
public:
	CrossEntropyNode(Graph *graph, int node, const std::vector<int> &labels) : Node{ node } {
		int size = (int)labels.size() * sizeof(int);
		int *labels_pinned = (int*)graph->GetDevice()->AllocateMemory(size, Device::PinnedScratchMemoryPool);
		memcpy(labels_pinned, labels.data(), size);
		labels_data_ = (int *)graph->GetDevice()->AllocateMemory(size, Device::ScratchMemoryPool);
		CUDA_CALL(cudaMemcpyAsync(labels_data_, labels_pinned, size, cudaMemcpyHostToDevice));
	}

	virtual Shape ForwardShape(const std::vector<Shape> &x_shapes) const override {
		Shape shape = x_shapes[0];
		shape.SetDim(shape.GetDimCount() - 1, 1);
		return shape;
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetDimCount() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		CrossEntropyForward<<<blocksPerGrid, threadsPerBlock>>>(x_data, y_data, labels_data_, size, dim_size);
		CUDA_CALL(cudaGetLastError());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetDimCount() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
		const float *x_data = x[0]->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		CrossEntropyBackward<<<blocksPerGrid, threadsPerBlock>>>(
			x_data, labels_data_, dEdY_data, dEdX_data, size, dim_size);
		CUDA_CALL(cudaGetLastError());
	}

private:
	int *labels_data_;
};

template class CrossEntropyNode<void, GPU>;

static __global__ void ClassificationAccuracyKernel(const float *input, const int *expected, float *output,
	int batch_size, int size) {
	int batch_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (batch_id < batch_size) {
		int max_index = 0;
		float max_value = input[batch_id * size];
		for (int i = 1; i < size; i++) {
			float current = input[batch_id * size + i];
			if (current > max_value) {
				max_value = current;
				max_index = i;
			}
		}
		if (max_index == expected[batch_id])
			output[batch_id] = 1.f;
		else
			output[batch_id] = 0.f;
	}
}

template<typename Dummy>
class ClassificationAccuracyNode<Dummy, GPU> : public Node {
public:
	ClassificationAccuracyNode(Graph *graph, int node, const std::vector<int> &labels) : Node{ node } {
		int size = (int)labels.size() * sizeof(int);
		int *labels_pinned = (int*)graph->GetDevice()->AllocateMemory(size, Device::PinnedScratchMemoryPool);
		memcpy(labels_pinned, labels.data(), size);
		labels_data_ = (int *)graph->GetDevice()->AllocateMemory(size, Device::ScratchMemoryPool);
		CUDA_CALL(cudaMemcpyAsync(labels_data_, labels_pinned, size, cudaMemcpyHostToDevice));
	}

	virtual Shape ForwardShape(const std::vector<Shape> &x_shapes) const override {
		Shape shape = x_shapes[0];
		shape.SetDim(shape.GetDimCount() - 1, 1);
		return shape;
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetDimCount() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetDimCount() - 1);
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		ClassificationAccuracyKernel<<<blocksPerGrid, threadsPerBlock>>>(
			x_data, labels_data_, y_data, size, dim_size);
		CUDA_CALL(cudaGetLastError());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		abort();
	}

private:
	int *labels_data_;
};

template class ClassificationAccuracyNode<void, GPU>;

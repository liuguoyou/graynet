#include "Device.h"
#include "Graph.h"
#include "Expression.h"
#include "Expression_p.h"
#include "Node.h"
#include "Utils.h"

#include <type_traits>
#include <cudnn.h>
#include <cub/block/block_reduce.cuh>

static const int kThreadsPerBlock = 128;
static const int kMaxThreadsPerBlock = 512;

static inline __device__ int GetTensorStorageIndex(int logical_index, int ndims, const int *elems, const int *strides) {
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

struct ReduceDesc {
	int regular_sizes[kMaxTensorDim + 1], reduce_sizes[kMaxTensorDim + 1];
	int strides[kMaxTensorDim + 1];
};

// Reduction modes:
// 0. No reduction : each thread handle one output element from one input element.
// 1. Small reduction : reduction size is less than kSmallReducesInBlock, each thread do one reduction. (TODO)
// 2. Medium reduction : reduction size is less than kMaxThreadsPerBlock, each warp do one reduction (TODO)
// 3. Large reduction : reduction size is less than kMaxThreadsPerBlock * kMaxReducePerThread,
// each thread block do one reduction.
// 4. Huge reduction : reduction size is larger than kMaxThreadsPerBlock * kMaxReducePerThread,
// reduce is distributed over several blocks. (TODO, kMaxReducePerThread is currently set to 2147483647).

//static const int kSmallReducesInBlock = 32;
//static const int kMaxSmallReductionSize = kMaxThreadsPerBlock / kSmallReducesInBlock;
static const int kMaxReducePerThread = 2147483647;

template<typename TransformFunc, typename StoreFunc, typename ExtraData>
static __global__ void TransformReduceKernel(TransformFunc transform_func, StoreFunc store_func,
	int dims, int regular_total, ExtraData extra_data) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < regular_total) {
		float value = transform_func(index, extra_data);
		store_func(index, value, extra_data);
	}
}

template<typename TransformFunc, typename ReduceFunc, typename StoreFunc, typename ExtraData>
static __global__ void TransformReduceKernel(TransformFunc transform_func, ReduceFunc reduce_func, StoreFunc store_func,
	int dims, int regular_total, int reduce_total, ReduceDesc reduce_desc, int reduces_per_thread, ExtraData extra_data) {
	typedef cub::BlockReduce<float, kThreadsPerBlock> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int regular_idx = blockIdx.x;
	int reduce_idx_base = threadIdx.x * reduces_per_thread;
	int base_idx = GetTensorStorageIndex(regular_idx, dims, reduce_desc.regular_sizes, reduce_desc.strides);
	// First element
	int index = base_idx + GetTensorStorageIndex(reduce_idx_base, dims, reduce_desc.reduce_sizes, reduce_desc.strides);
	float value = transform_func(index, extra_data);
	for (int reduce_idx = reduce_idx_base + 1; reduce_idx < reduce_idx_base + reduces_per_thread; reduce_idx++) {
		if (reduce_idx < reduce_total) {
			int index = base_idx + GetTensorStorageIndex(reduce_idx, dims, reduce_desc.reduce_sizes, reduce_desc.strides);
			float cur_value = transform_func(index, extra_data);
			// Reduce element
			value = reduce_func(value, cur_value);
		}
	}

	float result = BlockReduceT(temp_storage).Reduce(value, reduce_func, reduce_total);
	if (threadIdx.x == 0)
		store_func(base_idx, result, extra_data);
}

template<typename TransformFunc, typename ReduceFunc, typename StoreFunc, typename ExtraData>
static void TransformReduce(TransformFunc transform_func, ReduceFunc reduce_func, StoreFunc store_func,
	int dims, int regular_total, int regular_sizes[kMaxTensorDim + 1],
	int reduce_total, int reduce_sizes[kMaxTensorDim + 1], int strides[kMaxTensorDim + 1],
	const ExtraData &extra_data) {

	ReduceDesc desc;
	memcpy(&desc.regular_sizes, regular_sizes, sizeof(desc.regular_sizes));
	memcpy(&desc.reduce_sizes, reduce_sizes, sizeof(desc.reduce_sizes));
	memcpy(&desc.strides, strides, sizeof(desc.strides));
	
	if (reduce_total == 1) {
		// 0. No reduction
		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (regular_total + threadsPerBlock - 1) / threadsPerBlock;
		
		TransformReduceKernel<<<blocksPerGrid, threadsPerBlock>>>(transform_func, store_func,
			dims, regular_total, extra_data);
	}
	else {
		// 3. Large reduction
		int reduces_per_thread = (reduce_total + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
		if (reduces_per_thread > kMaxReducePerThread)
			DEBUG_BREAK(); // TODO

		int blocksPerGrid = regular_total;
		int threadsPerBlock;
		if (reduce_total < kMaxThreadsPerBlock)
			threadsPerBlock = reduce_total;
		else
			threadsPerBlock = kMaxThreadsPerBlock;

		TransformReduceKernel<<<blocksPerGrid, threadsPerBlock>>>(transform_func, reduce_func, store_func,
			dims, regular_total, reduce_total, desc, reduces_per_thread, extra_data);
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

template<typename DimFunc>
static void GetReduceDims(DimFunc dim_func, int dims, int *regular_total, int *reduce_total,
	int regular_sizes[kMaxTensorDim + 1], int reduce_sizes[kMaxTensorDim + 1], int strides[kMaxTensorDim + 1]) {
	int regular_tot = 1, reduce_tot = 1;
	int tot = 1;
	for (int i = dims - 1; i >= 0; i--) {
		std::pair<int, int> dim_desc = dim_func(i);
		int from_dim = dim_desc.first, to_dim = dim_desc.second;
		strides[i] = tot;
		regular_sizes[i] = regular_tot;
		reduce_sizes[i] = reduce_tot;
		tot *= from_dim;
		if (from_dim == to_dim) {
			// Regular dimension
			regular_tot *= from_dim;
		}
		else if (to_dim == 1) {
			// Reduce dimension
			reduce_tot *= from_dim;
		}
		else // Invalid reduction operation
			DEBUG_BREAK();
	}
	*regular_total = regular_tot;
	*reduce_total = reduce_tot;
}

struct BinaryReduceDesc {
	int lhs_strides[kMaxTensorDim + 1], rhs_strides[kMaxTensorDim + 1];
	int strides[kMaxTensorDim + 1];
};

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
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value) {
			abort();
		}

		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdL_data = dEdX[0]->GetData(), *dEdR_data = dEdX[1]->GetData();
		const Shape &lhs_shape = x[0]->GetShape(), &rhs_shape = x[1]->GetShape();
		const Shape &y_shape = y->GetShape();
		int ndims = 1 + y_shape.GetDimCount();

		int lhs_strides[kMaxTensorDim + 1], rhs_strides[kMaxTensorDim + 1];
		GetTensorStrides(x[0], lhs_strides);
		GetTensorStrides(x[1], rhs_strides);

		int regular_total, reduce_total;
		int regular_sizes[kMaxTensorDim + 1], reduce_sizes[kMaxTensorDim + 1];
		int strides[kMaxTensorDim + 1];
		BinaryReduceDesc desc;

		/* LHS */
		{
			GetReduceDims([&](int dim) {
				return dim == 0 ? std::make_pair(y->GetBatchSize(), x[0]->GetBatchSize())
					: std::make_pair(y_shape.GetDim(dim - 1), lhs_shape.GetDim(dim - 1));
			}, ndims, &regular_total, &reduce_total, regular_sizes, reduce_sizes, strides);

			memcpy(&desc.lhs_strides, lhs_strides, sizeof(desc.lhs_strides));
			memcpy(&desc.rhs_strides, rhs_strides, sizeof(desc.rhs_strides));
			memcpy(&desc.strides, strides, sizeof(desc.strides));
			auto transform_func = [=] __device__(int index, const BinaryReduceDesc &desc) {
				int lhs_index = GetTensorStorageIndex(index, ndims, desc.strides, desc.lhs_strides);
				int rhs_index = GetTensorStorageIndex(index, ndims, desc.strides, desc.rhs_strides);
				float dYdL_value, dYdR_value;
				BackwardFunc()(lhs_data[lhs_index], rhs_data[rhs_index], y_data[index], &dYdL_value, &dYdR_value);
				return dEdY_data[index] * dYdL_value;
			};
			auto store_func = [=] __device__(int index, float result, const BinaryReduceDesc &desc) {
				int lhs_index = GetTensorStorageIndex(index, ndims, desc.strides, desc.lhs_strides);
				dEdL_data[lhs_index] += result;
			};
			TransformReduce(transform_func, cub::Sum(), store_func,
				ndims, regular_total, regular_sizes, reduce_total, reduce_sizes, strides, desc);
		}

		/* RHS */
		{
			GetReduceDims([&](int dim) {
				return dim == 0 ? std::make_pair(y->GetBatchSize(), x[1]->GetBatchSize())
					: std::make_pair(y_shape.GetDim(dim - 1), rhs_shape.GetDim(dim - 1));
			}, ndims, &regular_total, &reduce_total, regular_sizes, reduce_sizes, strides);

			auto transform_func = [=] __device__(int index, const BinaryReduceDesc &desc) {
				int lhs_index = GetTensorStorageIndex(index, ndims, desc.strides, desc.lhs_strides);
				int rhs_index = GetTensorStorageIndex(index, ndims, desc.strides, desc.rhs_strides);
				float dYdL_value, dYdR_value;
				BackwardFunc()(lhs_data[lhs_index], rhs_data[rhs_index], y_data[index], &dYdL_value, &dYdR_value);
				return dEdY_data[index] * dYdR_value;
			};
			auto store_func = [=] __device__(int index, float result, const BinaryReduceDesc &desc) {
				int rhs_index = GetTensorStorageIndex(index, ndims, desc.strides, desc.rhs_strides);
				dEdR_data[rhs_index] += result;
			};
			TransformReduce(transform_func, cub::Sum(), store_func,
				ndims, regular_total, regular_sizes, reduce_total, reduce_sizes, strides, desc);
		}
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
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value) {
			abort();
		}

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
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value) {
			abort();
		}

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

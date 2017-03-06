#include "Device.h"
#include "Graph.h"
#include "Expression.h"
#include "Expression_p.h"
#include "Node.h"
#include "Utils.h"

#include <type_traits>
#include <cudnn.h>
#include <cub/block/block_reduce.cuh>
#include <curand.h>
#include <cusparse_v2.h>

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
	typedef cub::BlockReduce<float, kMaxThreadsPerBlock> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int regular_idx = blockIdx.x;
	int reduce_idx_base = threadIdx.x;
	int base_idx = GetTensorStorageIndex(regular_idx, dims, reduce_desc.regular_sizes, reduce_desc.strides);
	// First element
	int index = base_idx + GetTensorStorageIndex(reduce_idx_base, dims, reduce_desc.reduce_sizes, reduce_desc.strides);
	float value = transform_func(index, extra_data);
	int reduce_idx = reduce_idx_base;
	for (int i = 1; i < reduces_per_thread; i++) {
		reduce_idx += blockDim.x;
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

static void GetReduceDims(int dims, const int *from_dims, const int *to_dims,
	int *regular_total, int *reduce_total,
	int regular_sizes[kMaxTensorDim + 1], int reduce_sizes[kMaxTensorDim + 1], int strides[kMaxTensorDim + 1]) {
	int regular_tot = 1, reduce_tot = 1;
	int tot = 1;
	for (int i = dims - 1; i >= 0; i--) {
		int from_dim = from_dims[i], to_dim = to_dims[i];
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

static __global__ void LookupForwardKernel(int total, int emb_size, const int *indices,
	const float *x, float *y) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < total) {
		int j = i / emb_size;
		int k = i % emb_size;
		y[i] = x[indices[j] * emb_size + k];
	}
}

static __global__ void LookupBackwardKernel(int total, int emb_size, const int *indices,
	const float *dEdY, float *dEdX) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < total) {
		int j = i / emb_size;
		int k = i % emb_size;
		// TODO: Use a proper reduction mechanism, and try to make the reduction deterministic.
		atomicAdd(&dEdX[indices[j] * emb_size + k], dEdY[i]);
	}
}

class LookupNodeGPU : public Node {
public:
	LookupNodeGPU(Graph *graph, int embeddings, int batch_size, const Shape &shape, const int *indices)
		: Node{ embeddings }, batch_size_(batch_size), shape_(shape) {
		int size = batch_size * shape.GetSize() * sizeof(int);
		indices_pinned_ = (int*)graph->GetDevice()->AllocateMemoryPinned(size);
		memcpy(indices_pinned_, indices, size);
		indices_ = (int *)graph->GetDevice()->AllocateMemory(size);
		CUDA_CALL(cudaMemcpyAsync(indices_, indices_pinned_, size, cudaMemcpyHostToDevice));
	}

	virtual void FreeMemory(Device *device) {
		device->FreeMemoryPinned(indices_pinned_);
		device->FreeMemory(indices_);
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		int total = y->GetBatchSize() * y->GetShape().GetSize();
		int emb_size = y->GetShape().GetDim(1);

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
		LookupForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(total, emb_size, indices_,
			x_data, y_data);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int total = y->GetBatchSize() * y->GetShape().GetSize();
		int emb_size = y->GetShape().GetDim(1);
		
		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
		LookupBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(total, emb_size, indices_,
			dEdY_data, dEdX_data);
	}

private:
	int batch_size_;
	Shape shape_;
	int *indices_pinned_, *indices_;
};

template<typename Dummy>
struct LookupNodeFactory<Dummy, GPU> {
	Node *Create(Graph *graph, int embeddings, int batch_size, const Shape &shape, const int *indices) {
		return new LookupNodeGPU(graph, embeddings, batch_size, shape, indices);
	}
};

template struct LookupNodeFactory<void, GPU>;

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

struct BinaryReduceDesc {
	int lhs_strides[kMaxTensorDim + 1], rhs_strides[kMaxTensorDim + 1];
	int strides[kMaxTensorDim + 1];
};

template<typename ForwardFunc, typename BackwardFunc>
class BinaryOpNodeGPU : public Node {
public:
	BinaryOpNodeGPU(int lhs_node, int rhs_node) : Node{ lhs_node, rhs_node } {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		int size = y->GetShape().GetSize();
		float *y_data = y->GetData();
		int y_batch_size = y->GetBatchSize();
		const Shape &y_shape = y->GetShape();

		int nelems = y_batch_size * y_shape.GetSize();
		int ndims = 1 + y_shape.GetRank();
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
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value)
			REPORT_ERROR("Backward propagation is unsupported for this expression.");

		const float *lhs_data = x[0]->GetData(), *rhs_data = x[1]->GetData();
		const float *y_data = y->GetData();
		const float *dEdY_data = dEdY->GetData();
		float *dEdL_data = dEdX[0]->GetData(), *dEdR_data = dEdX[1]->GetData();
		const Shape &lhs_shape = x[0]->GetShape(), &rhs_shape = x[1]->GetShape();
		const Shape &y_shape = y->GetShape();
		int ndims = 1 + y_shape.GetRank();

		int lhs_dims[kMaxTensorDim + 1], rhs_dims[kMaxTensorDim + 1];
		int y_dims[kMaxTensorDim + 1];
		GetTensorDims(x[0], lhs_dims);
		GetTensorDims(x[1], rhs_dims);
		GetTensorDims(y, y_dims);

		int lhs_strides[kMaxTensorDim + 1], rhs_strides[kMaxTensorDim + 1];
		GetTensorStrides(x[0], lhs_strides);
		GetTensorStrides(x[1], rhs_strides);

		int regular_total, reduce_total;
		int regular_sizes[kMaxTensorDim + 1], reduce_sizes[kMaxTensorDim + 1];
		int strides[kMaxTensorDim + 1];
		BinaryReduceDesc desc;

		/* LHS */
		{
			GetReduceDims(ndims, y_dims, lhs_dims, 
				&regular_total, &reduce_total, regular_sizes, reduce_sizes, strides);

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
			GetReduceDims(ndims, y_dims, rhs_dims,
				&regular_total, &reduce_total, regular_sizes, reduce_sizes, strides);

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

template<typename ForwardFunc, typename BackwardFunc>
struct BinaryOpNodeFactory<GPU, ForwardFunc, BackwardFunc> {
	Node *Create(int lhs_node, int rhs_node) {
		return new BinaryOpNodeGPU<ForwardFunc, BackwardFunc>(lhs_node, rhs_node);
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
class BinaryLeftScalarOpNodeGPU : public Node {
public:
	BinaryLeftScalarOpNodeGPU(float lhs_scalar, int rhs_node) : Node{ rhs_node }, lhs_scalar_(lhs_scalar) {}

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
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value)
			REPORT_ERROR("Backward propagation is unsupported for this expression.");

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

template<typename ForwardFunc, typename BackwardFunc>
struct BinaryLeftScalarOpNodeFactory<GPU, ForwardFunc, BackwardFunc> {
	Node *Create(float lhs_scalar, int rhs_node) {
		return new BinaryLeftScalarOpNodeGPU<ForwardFunc, BackwardFunc>(lhs_scalar, rhs_node);
	}
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
class BinaryRightScalarOpNodeGPU : public Node {
public:
	BinaryRightScalarOpNodeGPU(int lhs_node, float rhs_scalar) : Node{ lhs_node }, rhs_scalar_(rhs_scalar) {}
	
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
		if (std::is_same<BackwardFunc, BinaryNoBackward>::value)
			REPORT_ERROR("Backward propagation is unsupported for this expression.");

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

template<typename ForwardFunc, typename BackwardFunc>
struct BinaryRightScalarOpNodeFactory<GPU, ForwardFunc, BackwardFunc> {
	Node *Create(int lhs_node, float rhs_scalar) {
		return new BinaryRightScalarOpNodeGPU<ForwardFunc, BackwardFunc>(lhs_node, rhs_scalar);
	}
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
class UnaryOpNodeGPU : public Node {
public:
	UnaryOpNodeGPU(int node) : Node{ node } {}

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

template<typename ForwardFunc, typename BackwardFunc>
struct UnaryOpNodeFactory<GPU, ForwardFunc, BackwardFunc> {
	Node *Create(int node) {
		return new UnaryOpNodeGPU<ForwardFunc, BackwardFunc>(node);
	}
};

INSTANTIATE_BINARY_OPS(GPU)
INSTANTIATE_BINARY_LEFT_SCALAR_OPS(GPU)
INSTANTIATE_BINARY_RIGHT_SCALAR_OPS(GPU)
INSTANTIATE_UNARY_OPS(GPU)

class SparseDotNodeGPU : public Node {
public:
	SparseDotNodeGPU(int lhs, int rhs) : Node{ lhs, rhs } {
		CUSPARSE_CALL(cusparseCreateMatDescr(&mat_desc_));
		CUSPARSE_CALL(cusparseSetMatType(mat_desc_, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_CALL(cusparseSetMatIndexBase(mat_desc_, CUSPARSE_INDEX_BASE_ZERO));
	}

	virtual ~SparseDotNodeGPU() {
		CUSPARSE_CALL(cusparseDestroyMatDescr(mat_desc_));
	}

	virtual int GetFlags() const override {
		return NoAllocateBackwardOutput;
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Tensor *lhs = x[0], *rhs = x[1];
		float alpha = 1.f, beta = 0.f;
		CUSPARSE_CALL(cusparseScsrmv(graph->GetDevice()->GetCuSPARSEHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
			lhs->GetBatchSize(), lhs->GetShape().GetDim(0), lhs->GetNonZeroCount(),
			&alpha, mat_desc_, lhs->GetSparseData(), lhs->GetSparseRowIndices(), lhs->GetSparseColumnIndices(),
			rhs->GetData(), &beta, y->GetData()));
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const Tensor *lhs = x[0], *rhs = x[1];
		Tensor *dEdL = dEdX[0], *dEdR = dEdX[1];
		AllocateClearTensor(graph, dEdR);
		// dEdL += dEdY * R'
		// dEdR += L' * dEdY
		float alpha = 1.f, beta = 1.f;
		// dEdL not implemented for now.
		CUSPARSE_CALL(cusparseScsrmv(graph->GetDevice()->GetCuSPARSEHandle(), CUSPARSE_OPERATION_TRANSPOSE,
			lhs->GetBatchSize(), lhs->GetShape().GetDim(0), lhs->GetNonZeroCount(),
			&alpha, mat_desc_, lhs->GetSparseData(), lhs->GetSparseRowIndices(), lhs->GetSparseColumnIndices(),
			dEdY->GetData(), &beta, dEdR->GetData()));
	}

private:
	cusparseMatDescr_t mat_desc_;
};

template<typename Dummy>
struct SparseDotNodeFactory<Dummy, GPU> {
	Node *Create(int lhs_node, int rhs_node) {
		return new SparseDotNodeGPU(lhs_node, rhs_node);
	}
};

template struct SparseDotNodeFactory<void, GPU>;

class PoolingNodeGPU : public Node {
public:
	PoolingNodeGPU(int node, const Shape &filter_shape, const Shape &strides, const Shape &padding, PoolingMode mode)
		: Node{ node }, filter_shape_(filter_shape), strides_(strides), padding_(padding), mode_(mode) {
		CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc_));
		CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_));
		CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_));
	}

	virtual ~PoolingNodeGPU() {
		CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_desc_));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc_));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc_));
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &x_shape = x[0]->GetShape();
		const Shape &y_shape = y->GetShape();
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		int ndims = y->GetShape().GetRank() - 1;

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
		if (mode_ == PoolingMode::MaxPooling)
			pooling_mode = CUDNN_POOLING_MAX;
		else if (mode_ == PoolingMode::AvgPooling)
			pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
		else if (mode_ == PoolingMode::AvgPoolingWithPadding)
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

		float alpha = 1.f, beta = 1.f;
		CUDNN_CALL(cudnnPoolingBackward(graph->GetDevice()->GetCuDNNHandle(), pooling_desc_,
			&alpha, y_desc_, y_data, y_desc_, dEdY_data, x_desc_, x_data, &beta,
			x_desc_, dEdX_data));
	}

private:
	Shape filter_shape_, strides_, padding_;
	PoolingMode mode_;
	cudnnPoolingDescriptor_t pooling_desc_;
	cudnnTensorDescriptor_t x_desc_, y_desc_;
};

template<typename Dummy>
struct PoolingNodeFactory<Dummy, GPU> {
	Node *Create(int node, const Shape &filter_shape, const Shape &strides, const Shape &padding, PoolingMode mode) {
		return new PoolingNodeGPU(node, filter_shape, strides, padding, mode);
	}
};

template struct PoolingNodeFactory<void, GPU>;

struct ReduceSumDesc {
	int x_strides[kMaxTensorDim + 1], y_strides[kMaxTensorDim + 1];
};

struct Empty {};

static __global__ void ReduceSumBackwardKernel(int nelems, int size,
	const float *dEdY, float *dEdX) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nelems) {
		dEdX[i] += dEdY[i / size];
	}
}

class ReduceSumNodeGPU : public Node {
public:
	ReduceSumNodeGPU(int node, int axis) : Node{ node }, axis_(axis) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *x_data = (float*)x[0]->GetData();
		float *y_data = (float*)y->GetData();

		int size = x[0]->GetShape().GetSize();
		int x_dims[kMaxTensorDim + 1], y_dims[kMaxTensorDim + 1];
		int dims;
		ReduceSumDesc desc;
		if (axis_ == -1) {
			dims = 2;
			x_dims[0] = x[0]->GetBatchSize();
			x_dims[1] = x[0]->GetShape().GetSize();
			y_dims[0] = x[0]->GetBatchSize();
			y_dims[1] = 1;
			desc.y_strides[0] = 1;
			desc.y_strides[1] = 0;
		}
		else {
			dims = y->GetShape().GetRank() + 1;
			GetTensorDims(x[0], x_dims);
			GetTensorDims(y, y_dims);
			GetTensorStrides(y, desc.y_strides);
		}
		int regular_total, reduce_total;
		int regular_sizes[kMaxTensorDim + 1], reduce_sizes[kMaxTensorDim + 1];
		int strides[kMaxTensorDim + 1];
		GetReduceDims(dims, x_dims, y_dims, &regular_total, &reduce_total,
			regular_sizes, reduce_sizes, strides);

		memcpy(desc.x_strides, strides, sizeof(strides));

		auto transform_func = [=] __device__ (int index, Empty) {
			return x_data[index];
		};
		auto store_func = [=] __device__ (int index, float value, Empty) {
			int y_index = GetTensorStorageIndex(index, dims, desc.x_strides, desc.y_strides);
			y_data[y_index] = value;
		};
		TransformReduce(transform_func, cub::Sum(), store_func, dims,
			regular_total, regular_sizes, reduce_total, reduce_sizes, strides, Empty());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = (float*)dEdY->GetData();
		float *dEdX_data = (float*)dEdX[0]->GetData();

		int size = x[0]->GetShape().GetSize();
		int batch_size = x[0]->GetBatchSize();
		int nelems = size * batch_size;

		if (axis_ != -1)
			REPORT_ERROR("Unsupported.");

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (nelems + threadsPerBlock - 1) / threadsPerBlock;
		ReduceSumBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
			nelems, size, dEdY_data, dEdX_data);
	}

private:
	int axis_;
};

template<typename Dummy>
struct ReduceSumNodeFactory<Dummy, GPU> {
	Node *Create(int node, int axis) {
		return new ReduceSumNodeGPU(node, axis);
	}
};

template struct ReduceSumNodeFactory<void, GPU>;

struct SliceDesc {
	int elems[kMaxTensorDim + 1], strides[kMaxTensorDim + 1];
};

static __global__ void SliceForwardKernel(int count, int base_index, int ndims, SliceDesc desc,
	const float *x, float *y) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < count) {
		int index = base_index + GetTensorStorageIndex(i, ndims, desc.elems, desc.strides);
		y[i] = x[index];
	}
}

static __global__ void SliceBackwardKernel(int count, int base_index, int ndims, SliceDesc desc,
	const float *dEdY, float *dEdX) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < count) {
		int index = base_index + GetTensorStorageIndex(i, ndims, desc.elems, desc.strides);
		dEdX[index] += dEdY[i];
	}
}

class SliceNodeGPU : public Node {
public:
	SliceNodeGPU(int node, const Shape &start, const Shape &size) : Node{ node }, start_(start), size_(size) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		int ndims = x[0]->GetShape().GetRank() + 1;
		GetTensorStrides(x[0], desc_.strides);
		base_index_ = 0;
		for (int i = 1; i < ndims; i++)
			base_index_ += desc_.strides[i] * start_.GetDim(i - 1);
		desc_.elems[ndims - 1] = 1;
		for (int i = ndims - 2; i >= 0; i--)
			desc_.elems[i] = desc_.elems[i + 1] * size_.GetDim(i);
		count_ = desc_.elems[0] * x[0]->GetBatchSize();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (count_ + threadsPerBlock - 1) / threadsPerBlock;
		SliceForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(count_, base_index_, ndims,
			desc_, x[0]->GetData(), y->GetData());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		int ndims = x[0]->GetShape().GetRank() + 1;
		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (count_ + threadsPerBlock - 1) / threadsPerBlock;
		SliceBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(count_, base_index_, ndims,
			desc_, dEdY->GetData(), dEdX[0]->GetData());
	}

private:
	Shape start_, size_;
	mutable SliceDesc desc_;
	mutable int count_, base_index_;
};

template<typename Dummy>
struct SliceNodeFactory<Dummy, GPU> {
	Node *Create(int node, const Shape &start, const Shape &size) {
		return new SliceNodeGPU(node, start, size);
	}
};

template struct SliceNodeFactory<void, GPU>;

class ConcatNodeGPU : public Node {
public:
	ConcatNodeGPU(initializer_view<Expression> values, int axis) : Node(values), axis_(axis) {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		REPORT_ERROR("Concat() is unsupported on GPU.");
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		REPORT_ERROR("Concat() is unsupported on GPU.");
	}

private:
	int axis_;
};

template<typename Dummy>
struct ConcatNodeFactory<Dummy, GPU> {
	Node *Create(initializer_view<Expression> values, int axis) {
		return new ConcatNodeGPU(values, axis);
	}
};

template struct ConcatNodeFactory<void, GPU>;

static __global__ void DropoutForwardKernel(int n, float p, float mul_scale,
	const float *probs, const float *x, float *y) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		if (probs[i] <= p)
			y[i] = 0.f;
		else
			y[i] = x[i] * mul_scale;
	}
}

static __global__ void DropoutBackwardKernel(int n, float p, float mul_scale,
	const float *probs, const float *dEdY, float *dEdX) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		if (probs[i] > p)
			dEdX[i] += dEdY[i] * mul_scale;
	}
}

class DropoutNodeGPU : public Node {
public:
	DropoutNodeGPU(int node, float p) : Node{ node }, p_(p) {}

	virtual void FreeMemory(Device *device) {
		device->FreeMemory(probs_);
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();
		int size = x[0]->GetBatchSize() * x[0]->GetShape().GetSize();
		probs_ = (float *)graph->GetDevice()->AllocateMemory(size * sizeof(float));

		curandGenerator_t generator = graph->GetDevice()->GetCuRANDGenerator();
		CURAND_CALL(curandGenerateUniform(generator, probs_, size));
		float scale = 1.f / (1.f - p_);

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		DropoutForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(size, p_, scale, probs_, x_data, y_data);
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		const float *dEdY_data = dEdY->GetData();
		float *dEdX_data = dEdX[0]->GetData();
		int size = x[0]->GetBatchSize() * x[0]->GetShape().GetSize();
		float scale = 1.f - p_;
		
		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		DropoutBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(size, p_, scale, probs_, dEdY_data, dEdX_data);
	}

private:
	float p_;
	mutable float *probs_;
};

template<typename Dummy>
struct DropoutNodeFactory<Dummy, GPU> {
	Node *Create(int node, float p) {
		return new DropoutNodeGPU(node, p);
	}
};

template struct DropoutNodeFactory<void, GPU>;

class SoftmaxNodeGPU : public Node {
public:
	SoftmaxNodeGPU(int node) : Node{ node } {}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		// y = exp(x_i) / sum(exp(x_i))
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetRank() - 1);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
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
		int size = x[0]->GetShape().GetSizeRange(0, input_shape.GetRank() - 1);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
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

template<typename Dummy>
struct SoftmaxNodeFactory<Dummy, GPU> {
	Node *Create(int node) {
		return new SoftmaxNodeGPU(node);
	}
};

template struct SoftmaxNodeFactory<void, GPU>;

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

class CrossEntropyNodeGPU : public Node {
public:
	CrossEntropyNodeGPU(Graph *graph, int node, const std::vector<int> &labels) : Node{ node } {
		int size = (int)labels.size() * sizeof(int);
		labels_pinned_ = (int*)graph->GetDevice()->AllocateMemoryPinned(size);
		memcpy(labels_pinned_, labels.data(), size);
		labels_data_ = (int *)graph->GetDevice()->AllocateMemory(size);
		CUDA_CALL(cudaMemcpyAsync(labels_data_, labels_pinned_, size, cudaMemcpyHostToDevice));
	}

	virtual void FreeMemory(Device *device) override {
		device->FreeMemoryPinned(labels_pinned_);
		device->FreeMemory(labels_data_);
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetRank() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
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
		int size = input_shape.GetSizeRange(0, input_shape.GetRank() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
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
	int *labels_pinned_, *labels_data_;
};

template<typename Dummy>
struct CrossEntropyNodeFactory<Dummy, GPU> {
	Node *Create(Graph *graph, int node, const std::vector<int> &labels) {
		return new CrossEntropyNodeGPU(graph, node, labels);
	}
};

template struct CrossEntropyNodeFactory<void, GPU>;

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

class ClassificationAccuracyNodeGPU : public Node {
public:
	ClassificationAccuracyNodeGPU(Graph *graph, int node, const std::vector<int> &labels) : Node{ node } {
		int size = (int)labels.size() * sizeof(int);
		// We use CUDA's automatic data migration feature since we only need the labels once
		labels_pinned_ = (int*)graph->GetDevice()->AllocateMemoryPinned(size);
		memcpy(labels_pinned_, labels.data(), size);
	}

	virtual void FreeMemory(Device *device) override {
		device->FreeMemoryPinned(labels_pinned_);
	}

	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const override {
		const Shape &input_shape = x[0]->GetShape();
		int size = input_shape.GetSizeRange(0, input_shape.GetRank() - 2);
		size *= x[0]->GetBatchSize();
		int dim_size = input_shape.GetDim(input_shape.GetRank() - 1);
		const float *x_data = x[0]->GetData();
		float *y_data = y->GetData();

		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		ClassificationAccuracyKernel<<<blocksPerGrid, threadsPerBlock>>>(
			x_data, labels_pinned_, y_data, size, dim_size);
		CUDA_CALL(cudaGetLastError());
	}

	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const override {
		REPORT_ERROR("Backward propagation is unsupported for this expression.");
	}

private:
	int *labels_pinned_;
};

template<typename Dummy>
struct ClassificationAccuracyNodeFactory<Dummy, GPU> {
	Node *Create(Graph *graph, int node, const std::vector<int> &labels) {
		return new ClassificationAccuracyNodeGPU(graph, node, labels);
	}
};

template struct ClassificationAccuracyNodeFactory<void, GPU>;

#include "Device.h"
#include "Utils.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <pmmintrin.h>
#include <xmmintrin.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <cusparse_v2.h>
#endif

// TODO: Make this a parameter
static const size_t kMinimumPoolSize = 64 * 1048576;

#pragma warning(disable:4146)
#define ALIGN_TO(size, align) (((size) + (align) - 1) & -(align))

/*! @private */
class MemoryPool final {
public:
	MemoryPool(DeviceType device_type, Device::MemoryPoolType pool_type):
		device_type_(device_type), pool_type_(pool_type) {
		// Make sure there is at least one pool to simplify further processing
		AllocateNewPool(kMinimumPoolSize);
	}

	~MemoryPool() {
		for (const Pool &pool : pools_) {
#ifdef USE_CUDA
			if (device_type_ == GPU) {
				if (pool_type_ == Device::PinnedScratchMemoryPool)
					CUDA_CALL(cudaFreeHost(pool.start_ptr));
				else
					CUDA_CALL(cudaFree(pool.start_ptr));
			}
			else
#endif
				free(pool.start_ptr);
		}
	}

	void *AllocateMemory(int size) {
		const size_t kAlignment = 16;
		// TODO: Performance improvement
		for (Pool &pool : pools_) {
			size_t current = ALIGN_TO(pool.current, kAlignment);
			if (current + size <= pool.capacity) {
				void *ptr = (char *)pool.start_ptr + current;
				pool.current = current + size;
				return ptr;
			}
		}
		AllocateNewPool(size);
		void *ptr = (char *)pools_.back().start_ptr;
		pools_.back().current = size;
		return ptr;
	}

	void Clear() {
		for (Pool &pool : pools_)
			pool.current = 0;
	}

private:
	void AllocateNewPool(int minimum_size) {
		size_t size = ALIGN_TO((size_t)minimum_size, kMinimumPoolSize);
		Pool pool;
#ifdef USE_CUDA
		if (device_type_ == GPU) {
			if (pool_type_ == Device::PinnedScratchMemoryPool)
				CUDA_CALL(cudaHostAlloc(&pool.start_ptr, size, cudaHostAllocDefault));
			else
				CUDA_CALL(cudaMalloc(&pool.start_ptr, size));
		}
		else
#endif
			pool.start_ptr = malloc(size);
		if (!pool.start_ptr)
			DEBUG_BREAK();
		pool.current = 0;
		pool.capacity = size;
		pools_.push_back(pool);
	}

	struct Pool {
		void *start_ptr;
		size_t current;
		size_t capacity;
	};
	std::vector<Pool> pools_;
	DeviceType device_type_;
	Device::MemoryPoolType pool_type_;
};

/*! @private */
class DevicePrivate {
public:
	DeviceType device_type_;
#ifdef USE_CUDA
	cublasHandle_t cublas_handle_;
	cusparseHandle_t cusparse_handle_;
	cudnnHandle_t cudnn_handle_;
	curandGenerator_t curand_generator_;

	MemoryPool *pinned_scratch_memory_pool_;
#endif

	MemoryPool *permanent_memory_pool_, *scratch_memory_pool_;
};

Device::Device() : Device(GPU) {
}

Device::Device(DeviceType device_type): d(new DevicePrivate()) {
#ifdef USE_CUDA
	d->device_type_ = device_type;
	if (d->device_type_ == GPU) {
		CUDA_CALL(cudaSetDevice(0));
		CUBLAS_CALL(cublasCreate_v2(&d->cublas_handle_));
		CUSPARSE_CALL(cusparseCreate(&d->cusparse_handle_));
		CUDNN_CALL(cudnnCreate(&d->cudnn_handle_));
		CURAND_CALL(curandCreateGenerator(&d->curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(d->curand_generator_,
			std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::high_resolution_clock::now().time_since_epoch()).count()));
		CURAND_CALL(curandSetGeneratorOrdering(d->curand_generator_, CURAND_ORDERING_PSEUDO_SEEDED));
	}
	d->pinned_scratch_memory_pool_ = new MemoryPool(d->device_type_, PinnedScratchMemoryPool);
#else
	d->device_type_ = CPU;
#endif
	d->permanent_memory_pool_ = new MemoryPool(d->device_type_, PermanentMemoryPool);
	d->scratch_memory_pool_ = new MemoryPool(d->device_type_, ScratchMemoryPool);

	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

Device::~Device() {
#ifdef USE_CUDA
	if (d->device_type_ == GPU) {
		CUBLAS_CALL(cublasDestroy_v2(d->cublas_handle_));
		CUDNN_CALL(cudnnDestroy(d->cudnn_handle_));
	}
	delete d->pinned_scratch_memory_pool_;
#endif
	delete d->permanent_memory_pool_;
	delete d->scratch_memory_pool_;
	delete d;
}

DeviceType Device::GetDeviceType() const {
	return d->device_type_;
}

#ifdef USE_CUDA
cublasHandle_t Device::GetCuBLASHandle() const {
	return d->cublas_handle_;
}

cusparseHandle_t Device::GetCuSPARSEHandle() const {
	return d->cusparse_handle_;
}

cudnnHandle_t Device::GetCuDNNHandle() const {
	return d->cudnn_handle_;
}

curandGenerator_t Device::GetCuRANDGenerator() const {
	return d->curand_generator_;
}

#endif

void *Device::AllocateMemory(int size, MemoryPoolType memory_pool) {
	if (memory_pool == PermanentMemoryPool)
		return d->permanent_memory_pool_->AllocateMemory(size);
	else if (memory_pool == ScratchMemoryPool)
		return d->scratch_memory_pool_->AllocateMemory(size);
#ifdef USE_CUDA
	else if (memory_pool = PinnedScratchMemoryPool)
		return d->pinned_scratch_memory_pool_->AllocateMemory(size);
#endif
	else
		REPORT_ERROR("Invalid memory pool: %d", memory_pool);
}

void Device::ZeroMemory(void *ptr, int size) {
#ifdef USE_CUDA
	if (d->device_type_ == GPU)
		CUDA_CALL(cudaMemsetAsync(ptr, 0, size));
	else
#endif
		memset(ptr, 0, size);
}

void Device::CopyMemory(void *dst, const void *src, int size) {
#ifdef USE_CUDA
	if (d->device_type_ == GPU)
		CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice));
	else
#endif
		memcpy(dst, src, size);
}

void Device::ClearMemoryPool(MemoryPoolType memory_pool) {
	if (memory_pool == PermanentMemoryPool)
		d->permanent_memory_pool_->Clear();
	else if (memory_pool == ScratchMemoryPool)
		d->scratch_memory_pool_->Clear();
#ifdef USE_CUDA
	else if (memory_pool == PinnedScratchMemoryPool)
		d->pinned_scratch_memory_pool_->Clear();
#endif
	else
		REPORT_ERROR("Invalid memory pool: %d", memory_pool);
}

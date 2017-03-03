#include "Device.h"
#include "Utils.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <unordered_map>
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

/*! \private */
enum class AllocatorType {
	CPU,
	CPUPinned,
	GPU,
};

/*! \private */
/* A caching allocator for CPU and CUDA.
 * Memory allocations are categorized as small allocations (<=1MB) and large
 * allocations (>1MB). These two allocations use two different allocation pools,
 * which use the same protocol but different alignment settings.
 *
 * A memory allocation is fulfilled by finding a cached block with least size in
 * its corresponding memory pool. If such a block is found, the block is used.
 *
 * If a cached block cannot be found, for small allocations a new block of 1MB
 * will be created. For large allocations a new block of 64MB will be created. For
 * even larger allocations, a block with requested size, aligned to 1MB will be
 * created.
 *
 * A used block may be split if the remaining unused size is too large for the
 * corresponding memory pool.
 *
 * If an allocation fails, the allocator will free all cached blocks that are not
 * split and retry the allocation.
 */
template<AllocatorType type>
class CachingAllocator final {
public:
	~CachingAllocator() {
		if (!allocated_blocks_.empty()) {
			if (type == AllocatorType::CPU)
				printf("CPU Allocator: Memory leak detected.\n");
			else if (type == AllocatorType::CPUPinned)
				printf("CPU Pinned Allocator: Memory leak detected.\n");
			else if (type == AllocatorType::GPU)
				printf("GPU Allocator: Memory leak detected.\n");
		}
		for (CachedBlock *block : cached_blocks_)
			if (block->prev == nullptr)
				FreeDeviceMemory(block->ptr);
	}
	
	void *AllocateMemory(size_t size) {
		bool is_small = (size <= kSmallAllocationSize);
		int align = is_small ? kSmallAllocationAlign : kLargeAllocationAlign;
		size = ALIGN_TO(size, align);

		CachedBlock key;
		key.ptr = nullptr;
		key.size = size;
		auto it = cached_blocks_.lower_bound(&key);
		CachedBlock *block;
		if (it != cached_blocks_.end()) {
			// Found suitable block
			block = *it;
			cached_blocks_.erase(it);
		}
		else {
			// Not found suitable block, create a new block
			size_t request_size;
			if (is_small)
				request_size = kSmallAllocationSize;
			else if (size <= kLargeAllocationSize)
				request_size = kLargeAllocationSize;
			else
				request_size = ALIGN_TO(size, kLargeAllocationAlign);
			block = AllocateBlock(request_size);
		}

		// Split the block if remaining unused size is too large
		if (block->size - size >= align) {
			CachedBlock *next_block = new CachedBlock();
			next_block->ptr = block->ptr + size;
			next_block->size = block->size - size;
			next_block->prev = block;
			next_block->next = block->next;
			if (block->next)
				block->next->prev = next_block;
			next_block->in_use = false;
			cached_blocks_.insert(next_block);

			block->next = next_block;
			block->size = size;
		}
		block->in_use = true;
		allocated_blocks_.emplace(block->ptr, block);
		return block->ptr;
	}

	void FreeMemory(void *ptr) {
		auto it = allocated_blocks_.find(ptr);
		if (it == allocated_blocks_.end())
			REPORT_ERROR("Pointer %p is not allocated in this allocator.", ptr);
		CachedBlock *block = it->second;
		allocated_blocks_.erase(it);
		// Merge adjacent unused blocks
		if (block->prev && !block->prev->in_use) {
			CachedBlock *prev_block = block->prev;
			cached_blocks_.erase(prev_block);
			prev_block->size += block->size;
			prev_block->next = block->next;
			if (block->next)
				block->next->prev = prev_block;
			delete block;
			block = prev_block;
		}
		if (block->next && !block->next->in_use) {
			CachedBlock *next_block = block->next;
			cached_blocks_.erase(next_block);
			block->size += next_block->size;
			block->next = next_block->next;
			if (block->next)
				block->next->prev = block;
			delete next_block;
		}
		block->in_use = false;
		cached_blocks_.insert(block);
	}

private:
	static constexpr size_t kSmallAllocationSize = 1'048'576; // 1MB
	static constexpr size_t kSmallAllocationAlign = 256; // 256 bytes
	static constexpr size_t kLargeAllocationSize = 67'108'864; // 64MB
	static constexpr size_t kLargeAllocationAlign = 1'048'576; // 1MB
	// Ensures large and small cached blocks will never interfere
	static_assert(kLargeAllocationAlign >= kSmallAllocationSize, "");

	struct CachedBlock {
		// Memory address
		char *ptr;

		// Size
		size_t size;

		// Previous block if split from a larger allocation
		CachedBlock *prev;

		// Next block if split from a larger allocation
		CachedBlock *next;

		// Whether this block is in use
		bool in_use;
	};

	struct CachedBlockComparator {
		inline bool operator()(const CachedBlock *lhs, const CachedBlock *rhs) const {
			if (lhs->size != rhs->size)
				return lhs->size < rhs->size;
			else
				return lhs->ptr < rhs->ptr;
		}
	};

	std::unordered_map<void *, CachedBlock *> allocated_blocks_;
	std::set<CachedBlock *, CachedBlockComparator> cached_blocks_;

	CachedBlock *AllocateBlock(size_t size) {
		char *ptr = (char *)AllocateDeviceMemory(size);
		if (ptr == nullptr) {
			// Allocation failed, try to free all cached blocks that are not split
			for (auto it = cached_blocks_.begin(); it != cached_blocks_.end();) {
				CachedBlock *block = *it;
				if (block->prev == nullptr && block->next == nullptr) {
					FreeDeviceMemory(block->ptr);
					delete block;
					it = cached_blocks_.erase(it);
				}
				else
					++it;
			}
			// Now retry the allocation
			ptr = (char *)AllocateDeviceMemory(size);
			if (ptr == nullptr)
				REPORT_ERROR("Out of memory.");
		}
		CachedBlock *block = new CachedBlock();
		block->ptr = ptr;
		block->size = size;
		block->prev = nullptr;
		block->next = nullptr;
		block->in_use = false;
		return block;
	}

	void *AllocateDeviceMemory(size_t size) {
		void *ret = nullptr;
		if (type == AllocatorType::CPU)
			ret = malloc(size);
#ifdef USE_CUDA
		else if (type == AllocatorType::CPUPinned)
			cudaMallocHost(&ret, size);
		else if (type == AllocatorType::GPU)
			cudaMalloc(&ret, size);
#endif
		else
			DEBUG_BREAK();
		return ret;
	}

	void FreeDeviceMemory(void *ptr) {
		if (type == AllocatorType::CPU)
			free(ptr);
#ifdef USE_CUDA
		else if (type == AllocatorType::CPUPinned)
			CUDA_CALL(cudaFreeHost(ptr));
		else if (type == AllocatorType::GPU)
			CUDA_CALL(cudaFree(ptr));
#endif
		else
			DEBUG_BREAK();
	}
};

/*! \private */
class DevicePrivate {
public:
	DeviceType device_type_;
#ifdef USE_CUDA
	cublasHandle_t cublas_handle_;
	cusparseHandle_t cusparse_handle_;
	cudnnHandle_t cudnn_handle_;
	curandGenerator_t curand_generator_;

	CachingAllocator<AllocatorType::GPU> gpu_allocator_;
	CachingAllocator<AllocatorType::CPUPinned> pinned_allocator_;
#endif

	CachingAllocator<AllocatorType::CPU> cpu_allocator_;
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
#else
	d->device_type_ = CPU;
#endif

	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

Device::~Device() {
#ifdef USE_CUDA
	if (d->device_type_ == GPU) {
		CUBLAS_CALL(cublasDestroy_v2(d->cublas_handle_));
		CUDNN_CALL(cudnnDestroy(d->cudnn_handle_));
		CURAND_CALL(curandDestroyGenerator(d->curand_generator_));
	}
#endif
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

void *Device::AllocateMemory(size_t size) {
#ifdef USE_CUDA
	if (d->device_type_ == GPU)
		return d->gpu_allocator_.AllocateMemory(size);
	else
#endif
		return d->cpu_allocator_.AllocateMemory(size);
}

void Device::FreeMemory(void *ptr) {
#ifdef USE_CUDA
	if (d->device_type_ == GPU)
		d->gpu_allocator_.FreeMemory(ptr);
	else
#endif
		d->cpu_allocator_.FreeMemory(ptr);
}

void *Device::AllocateMemoryPinned(size_t size) {
#ifdef USE_CUDA
	if (d->device_type_ == GPU)
		return d->pinned_allocator_.AllocateMemory(size);
	else
#endif
		return d->cpu_allocator_.AllocateMemory(size);
}

void Device::FreeMemoryPinned(void *ptr) {
#ifdef USE_CUDA
	if (d->device_type_ == GPU)
		return d->pinned_allocator_.FreeMemory(ptr);
	else
#endif
		return d->cpu_allocator_.FreeMemory(ptr);
}

void Device::ZeroMemory(void *ptr, size_t size) {
#ifdef USE_CUDA
	if (d->device_type_ == GPU)
		CUDA_CALL(cudaMemsetAsync(ptr, 0, size));
	else
#endif
		memset(ptr, 0, size);
}

void Device::CopyMemory(void *dst, const void *src, size_t size) {
#ifdef USE_CUDA
	if (d->device_type_ == GPU)
		CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice));
	else
#endif
		memcpy(dst, src, size);
}

#pragma once

enum DeviceType {
	None,
	CPU,
	GPU,
};

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;
struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;
struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;
class DevicePrivate;
class Device {
public:
	Device();
	Device(DeviceType device_type);
	virtual ~Device();

	/*! Get device type */
	DeviceType GetDeviceType() const;

#ifdef USE_CUDA
	/*! \private */
	cublasHandle_t GetCuBLASHandle() const;
	/*! \private */
	cusparseHandle_t GetCuSPARSEHandle() const;
	/*! \private */
	cudnnHandle_t GetCuDNNHandle() const;
#endif

	/*! \private */
	enum MemoryPoolType {
		PermanentMemoryPool,
		ScratchMemoryPool,
		PinnedScratchMemoryPool,
	};
	/*! \private */
	void *AllocateMemory(int size, MemoryPoolType memory_pool);
	/*! \private */
	void ZeroMemory(void *ptr, int size);
	/*! \private */
	void CopyMemory(void *dst, const void *src, int size);
	/*! \private */
	void ClearMemoryPool(MemoryPoolType memory_pool);

private:
	DevicePrivate *d;
};

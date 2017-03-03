#pragma once

#include <cstddef>

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
struct curandGenerator_st;
typedef struct curandGenerator_st *curandGenerator_t;
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
	/*! \private */
	curandGenerator_t GetCuRANDGenerator() const;
#endif

	/*! \private */
	void *AllocateMemory(size_t size);
	/*! \private */
	void FreeMemory(void *ptr);
	/*! \private */
	void *AllocateMemoryPinned(size_t size);
	/*! \private */
	void FreeMemoryPinned(void *ptr);
	/*! \private */
	void ZeroMemory(void *ptr, size_t size);
	/*! \private */
	void CopyMemory(void *dst, const void *src, size_t size);

private:
	DevicePrivate *d;
};

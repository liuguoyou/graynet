#include "Tensor.h"
#include "Utils.h"

#include <cstdio>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

Tensor::Tensor(DeviceType device_type, const Shape &shape, float *data)
	: device_type_(device_type), batch_size_(1), shape_(shape), data_(data) {
}

Tensor::Tensor(DeviceType device_type, int batch_size, const Shape &shape, float *data)
	: device_type_(device_type), batch_size_(batch_size), shape_(shape), data_(data) {
}

float Tensor::GetValueFlat(int index) const {
#ifdef USE_CUDA
	if (device_type_ == GPU) {
		float value;
		CUDA_CALL(cudaMemcpy(&value, &data_[index], sizeof(float), cudaMemcpyDeviceToHost));
		return value;
	}
	else
#endif
		return data_[index];
}

void Tensor::SetValueFlat(int index, float value) {
#ifdef USE_CUDA
	if (device_type_ == GPU)
		CUDA_CALL(cudaMemcpy(&data_[index], &value, sizeof(float), cudaMemcpyHostToDevice));
	else
#endif
		data_[index] = value;
}

float Tensor::ToScalar() const {
#ifdef USE_CUDA
	if (device_type_ == GPU) {
		float data;
		CUDA_CALL(cudaMemcpy(&data, data_, sizeof(float), cudaMemcpyDeviceToHost));
		return data;
	}
	else
#endif
		return *data_;
}

float Tensor::ReduceSum() const {
	int total_size = batch_size_ * shape_.GetSize();
#ifdef USE_CUDA
	if (device_type_ == GPU) {
		float *dd = new float[total_size];
		CUDA_CALL(cudaMemcpy(dd, data_, total_size * sizeof(float), cudaMemcpyDeviceToHost));
		float sum = 0;
		for (int i = 0; i < total_size; i++)
			sum += dd[i];
		delete dd;
		return sum;
	}
	else
#endif
	{
		float sum = 0;
		for (int i = 0; i < total_size; i++)
			sum += data_[i];
		return sum;
	}
}

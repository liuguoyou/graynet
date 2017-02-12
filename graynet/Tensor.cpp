#include "Tensor.h"
#include "Utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

Tensor::Tensor(DeviceType device_type, const Shape &shape, float *data)
	: device_type_(device_type), is_sparse_(false), batch_size_(1), shape_(shape), data_(data) {
}

Tensor::Tensor(DeviceType device_type, int batch_size, const Shape &shape, float *data)
	: device_type_(device_type), is_sparse_(false), batch_size_(batch_size), shape_(shape), data_(data) {
}

Tensor::Tensor(DeviceType device_type, int batch_size, const Shape &shape, int nonzero_count,
	float *sparse_data, int *batch_indices, int *indices)
	: device_type_(device_type), is_sparse_(true), batch_size_(batch_size), shape_(shape),
	nonzero_count_(nonzero_count), data_(sparse_data), row_indices_(batch_indices), column_indices_(indices) {
}

float *Tensor::GetData() const {
	if (is_sparse_)
		abort();
	return data_;
}

void Tensor::GetValue(float *value) const {
	if (is_sparse_)
		abort();
	int size = batch_size_ * shape_.GetSize() * sizeof(float);
#ifdef USE_CUDA
	if (device_type_ == GPU)
		CUDA_CALL(cudaMemcpy(value, data_, size, cudaMemcpyDeviceToHost));
	else
#endif
		memcpy(value, data_, size);
}

float Tensor::GetValueFlat(int index) const {
	if (is_sparse_)
		abort();
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
	if (is_sparse_)
		abort();
#ifdef USE_CUDA
	if (device_type_ == GPU)
		CUDA_CALL(cudaMemcpy(&data_[index], &value, sizeof(float), cudaMemcpyHostToDevice));
	else
#endif
		data_[index] = value;
}

int Tensor::GetNonZeroCount() const {
	if (!is_sparse_)
		abort();
	return nonzero_count_;
}

float *Tensor::GetSparseData() const {
	if (!is_sparse_)
		abort();
	return data_;
}

int *Tensor::GetSparseColumnIndices() const {
	if (!is_sparse_)
		abort();
	return column_indices_;
}

int *Tensor::GetSparseRowIndices() const {
	if (!is_sparse_)
		abort();
	return row_indices_;
}

float Tensor::ToScalar() const {
	if (is_sparse_)
		abort();
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
	if (is_sparse_)
		abort();
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

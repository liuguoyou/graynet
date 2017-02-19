#pragma once

#include "Device.h"
#include "Shape.h"

/*! Pointer to CPU/GPU Tensor storage */
class Tensor {
public:
	/*! Construct an empty tensor object with given shape and batch size */
	Tensor(DeviceType device_type, int batch_size, const Shape &shape, bool is_sparse);

	/*! Construct a dense tensor object with given shape and data pointer. */
	Tensor(DeviceType device_type, const Shape &shape, float *data);

	/*! Construct a dense tensor object with given shape, batch size and data pointer. */
	Tensor(DeviceType device_type, int batch_size, const Shape &shape, float *data);

	/*! Construct a sparse vector tensor object with given shape, batch size and data pointers. */
	Tensor(DeviceType device_type, int batch_size, const Shape &shape, int nonzero_count,
		float *sparse_data, int *batch_indices, int *indices);

	/*! Test if this tensor does not contain data. */
	bool IsEmpty() const { return data_ == nullptr; }

	/*! Test if this tensor is a dense tensor. */
	bool IsDense() const { return !is_sparse_; }

	/*! Test if this tensor is a sparse tensor. */
	bool IsSparse() const { return is_sparse_; }

	/*! Get the batch size of the tensor. */
	int GetBatchSize() const { return batch_size_; }

	/*! Get the shape of the tensor. */
	const Shape &GetShape() const { return shape_; }

	/*! Get the data pointer of the tensor. */
	float *GetData() const;

	/*! Get tensor value back to provided CPU array */
	void GetValue(float *value) const;

	/*! Get value by flat index */
	float GetValueFlat(int index) const;

	/*! Set value by flat index */
	void SetValueFlat(int index, float value);

	/*! Get number of non zero data values. */
	int GetNonZeroCount() const;

	/*! Get the sparse data pointer of the tensor. */
	float *GetSparseData() const;

	/*! Get sparse column indices */
	int *GetSparseColumnIndices() const;

	/*! Get sparse row indices */
	int *GetSparseRowIndices() const;

	/*! To scalar value */
	float ToScalar() const;

	/*! Average scalar value over batches */
	float ReduceSum() const;

private:
	void EnsureDense() const;
	void EnsureSparse() const;

	DeviceType device_type_;
	bool is_sparse_;
	int batch_size_;
	Shape shape_;
	float *data_;
	int nonzero_count_;
	int *column_indices_;
	int *row_indices_;
};

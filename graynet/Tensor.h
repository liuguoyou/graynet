#pragma once

#include "Device.h"
#include "Shape.h"

/*! Pointer to CPU/GPU Tensor storage */
class Tensor {
public:
	/*! Construct a tensor object with given shape and data pointer. */
	Tensor(DeviceType device_type, const Shape &shape, float *data);

	/*! Construct a tensor object with given shape, batch size and data pointer. */
	Tensor(DeviceType device_type, int batch_size, const Shape &shape, float *data);

	/*! Get the batch size of the tensor. */
	int GetBatchSize() const { return batch_size_; }

	/*! Get the shape of the tensor. */
	const Shape &GetShape() const { return shape_; }

	/*! Get the data pointer of the tensor. */
	float *GetData() const { return data_; }

	/*! Get value by flat index */
	float GetValueFlat(int index) const;

	/*! Set value by flat index */
	void SetValueFlat(int index, float value);

	/*! To scalar value */
	float ToScalar() const;

	/*! Average scalar value over batches */
	float ReduceSum() const;

private:
	DeviceType device_type_;
	int batch_size_;
	Shape shape_;
	float *data_;
};

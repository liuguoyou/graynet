#pragma once

static const int kMaxTensorDim = 7;
class Shape {
public:
	Shape() : ndim_(0) {}
	Shape(int d1) : ndim_(1), dims_{ d1 } {}
	Shape(int d1, int d2) : ndim_(2), dims_{ d1, d2 } {}
	Shape(int d1, int d2, int d3) : ndim_(3), dims_{ d1, d2, d3 } {}
	Shape(int d1, int d2, int d3, int d4) : ndim_(4), dims_{ d1, d2, d3, d4 } {}
	Shape(int d1, int d2, int d3, int d4, int d5) : ndim_(5), dims_{ d1, d2, d3, d4, d5 } {}
	Shape(int d1, int d2, int d3, int d4, int d5, int d6) : ndim_(6), dims_{ d1, d2, d3, d4, d5, d6 } {}
	Shape(int d1, int d2, int d3, int d4, int d5, int d6, int d7) : ndim_(7), dims_{ d1, d2, d3, d4, d5, d6, d7 } {}
	
	inline int GetSize() const {
		int size = 1;
		for (int i = 0; i < ndim_; i++)
			size *= dims_[i];
		return size;
	}
	/*! Get size of dim range [begin_dim, end_dim). */
	inline int GetSizeRange(int begin_dim, int end_dim) const {
		int size = 1;
		for (int i = begin_dim; i < end_dim; i++)
			size *= dims_[i];
		return size;
	}
	inline int GetRank() const { return ndim_; }
	inline int GetDim(int dim) const { return dims_[dim]; }

	inline bool operator==(const Shape &rhs) const {
		for (int i = 0; i < ndim_; i++)
			if (dims_[i] != rhs.dims_[i])
				return false;
		return true;
	}

	inline bool operator!=(const Shape &rhs) const {
		for (int i = 0; i < ndim_; i++)
			if (dims_[i] == rhs.dims_[i])
				return false;
		return true;
	}

	const int *data() const {
		return dims_;
	}

	static Shape One(int ndims) {
		Shape shape;
		for (int i = 0; i < ndims; i++)
			shape.PushDim(1);
		return shape;
	}
	
	void DeleteDim(int dim_index) {
		for (int i = dim_index; i + 1 < ndim_; i++)
			dims_[i] = dims_[i + 1];
		ndim_--;
	}

	void SetDim(int dim_index, int new_size) {
		dims_[dim_index] = new_size;
	}

	void PushDim(int dim_value) {
		dims_[ndim_++] = dim_value;
	}

	void PushDims(const Shape &dims) {
		for (int i = 0; i < dims.GetRank(); i++)
			PushDim(dims.GetDim(i));
	}

private:
	int dims_[kMaxTensorDim];
	int ndim_;
};

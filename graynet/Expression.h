#pragma once

#include "Tensor.h"

class Graph;
class Shape;
class Expression {
public:
	/*! Default constructor, constructs an invalid expression handle. */
	Expression();

	/*! Test if this object is a valid expression */
	bool IsValid() const { return graph_ != nullptr; }

	/*! Get the graph object this expression belongs to */
	Graph *GetGraph() const { return graph_; }

	/*! Get shape of this expression */
	Shape GetShape() const;

	/*! Forward compute this expression, return its value */
	Tensor Forward() const;

	/*! Backward compute this expression, update gradient accumulation in the graph */
	void Backward() const;

	/*! @private */
	/*! Construct an expression from a graph and node index */
	Expression(Graph *graph, int index) : graph_(graph), index_(index) {}

	/*! @private */
	/*! Get node index in graph */
	int GetNodeIndex() const { return index_; }

private:
	Graph *graph_;
	int index_;
};

/* Graph Operators */

/*! Data input */
Expression Input(Graph *graph, const Shape &shape, const float *data);

/*! Batch data input */
Expression BatchInput(Graph *graph, int batch_size, const Shape &shape, const float *data);

/*! Batched sparse vector input (CSR format) */
Expression BatchSparseVectorInput(Graph *graph, int batch_size, const Shape &shape,
	int nonzero_count, const float *sparse_data, const int *batch_indices, const int *indices);

/*! Element-wise addition */
Expression operator+(const Expression &lhs, const Expression &rhs);

/*! Element-wise addition */
Expression operator+(float lhs, const Expression &rhs);

/*! Element-wise addition */
Expression operator+(const Expression &lhs, float rhs);

/*! Element-wise subtraction */
Expression operator-(const Expression &lhs, const Expression &rhs);

/*! Element-wise subtraction */
Expression operator-(float lhs, const Expression &rhs);

/*! Element-wise subtraction */
Expression operator-(const Expression &lhs, float rhs);

/*! Element-wise multiplication */
Expression operator*(const Expression &lhs, const Expression &rhs);

/*! Element-wise multiplication */
Expression operator*(float lhs, const Expression &rhs);

/*! Element-wise multiplication */
Expression operator*(const Expression &lhs, float rhs);

/*! Element-wise division */
Expression operator/(const Expression &lhs, const Expression &rhs);

/*! Element-wise division */
Expression operator/(float lhs, const Expression &rhs);

/*! Element-wise division */
Expression operator/(const Expression &lhs, float rhs);

/*! Element-wise negation: y = -x */
Expression operator-(const Expression &x);

/*! Element-wise square function: y = x^2 */
Expression Square(const Expression &x);

/*! Element-wise cube function: y = x^3 */
Expression Cube(const Expression &x);

/*! Element-wise exponential function: y = exp(x) */
Expression Exp(const Expression &x);

/*! Element-wise logarithm function: y = log(x) */
Expression Log(const Expression &x);

/*! Absolute value function: y = abs(x) */
Expression Abs(const Expression &x);

/*! Square root function: y = sqrt(x) */
Expression Sqrt(const Expression &x);

/*! Cubic root function: y = cbrt(x) */
Expression Cbrt(const Expression &x);

/*! Sine function: y = sin(x) */
Expression Sin(const Expression &x);

/*! Cosine function: y = cos(x) */
Expression Cos(const Expression &x);

/*! Tangent function: y = tan(x) */
Expression Tan(const Expression &x);

/*! Inverse sine function: y = asin(x) */
Expression Asin(const Expression &x);

/*! Inverse cosine function: y = acos(x) */
Expression Acos(const Expression &x);

/*! Inverse tangent function: y = actan(x) */
Expression Atan(const Expression &x);

/* Hyperbolic sine function: y = sinh(x) */
Expression Sinh(const Expression &x);

/* Hyperbolic cosine function: y = cosh(x) */
Expression Cosh(const Expression &h);

/* Hyperbolic tangent function: y = tanh(x) */
Expression Tanh(const Expression &x);

/*! Sigmoid function: y = 1/(1+e^-x) */
Expression Sigmoid(const Expression &x);

/*! ReLU function: y = max(x, 0) */
Expression ReLU(const Expression &x);

/*! Matrix multiplication */
Expression MatMul(const Expression &lhs, const Expression &rhs);

/*! Vector dot (only sparse dot is supported for now) */
Expression Dot(const Expression &lhs, const Expression &rhs);

/*! N-D Convolution (actually cross correlation)
 * x: CHW format
 * filter shape: [output_channels, input_channels, d...]
 * x shape: [input_channels, d...]
 */
Expression Convolution(const Expression &x, const Expression &filter, const Shape &strides, const Shape &padding);

/*! N-D Max Pooling */
Expression MaxPooling(const Expression &x, const Shape &filter_shape, const Shape &strides, const Shape &padding);

/*! Reshape tensor */
Expression Reshape(const Expression &x, const Shape &shape);

/*! Softmax function over the last dimension: y = exp(x_i) / sum(exp(x_i)) */
Expression Softmax(const Expression &x);

/*! Soft margin (logistic) loss
 * y = ln(1 + e^-(label*x))
 * label must be -1 or 1
 */
Expression SoftMargin(const Expression &x, const Expression &label);

/*! Binary cross entropy loss
 * y = -label*ln(x) - (1-label)*ln(1-x)
 * label must be 0 or 1
 */
Expression BinaryCrossEntropy(const Expression &x, const Expression &label);

/*! Binary classification accuracy: y = (x > 0.5 == label > 0.5) */
Expression BinaryClassificationAccuracy(const Expression &x, const Expression &label);

/*! Cross entropy loss: y = -log(x_k) */
Expression CrossEntropy(const Expression &x, int size, const int *labels);

/*! Classification accuracy: y = (argmax(x) == label) */
Expression ClassificationAccuracy(const Expression &x, int size, const int *labels);

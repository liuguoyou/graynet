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

/*! Element-wise add */
Expression operator+(const Expression &lhs, const Expression &rhs);

/*! Element-wise subtract */
Expression operator-(const Expression &lhs, const Expression &rhs);

/*! Element-wise multiply */
Expression ElemMul(const Expression &lhs, const Expression &rhs);

/*! Element-wise divide */
Expression ElemDiv(const Expression &lhs, const Expression &rhs);

/*! Element-wise negation: y = -x */
Expression operator-(const Expression &x);

/*! Element-wise square function: y = x^2 */
Expression Sqr(const Expression &x);

/*! Element-wise cube function: y = x^3 */
Expression Cube(const Expression &x);

/*! Element-wise exponential function: y = exp(x) */
Expression Exp(const Expression &x);

/*! Sigmoid function: y = 1/(1+e^-x) */
Expression Sigmoid(const Expression &x);

/*! Tanh function: y = tanh(x) */
Expression Tanh(const Expression &x);

/*! ReLU function: y = max(x, 0) */
Expression ReLU(const Expression &x);

/*! Matrix multiplication */
Expression operator*(const Expression &lhs, const Expression &rhs);

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

/*! Binary classification accuracy: y = (x == label) */
Expression BinaryClassificationAccuracy(const Expression &x, const Expression &label);

/*! Cross entropy loss: y = -log(x_k) */
Expression CrossEntropy(const Expression &x, int size, const int *labels);

/*! Classification accuracy: y = (argmax(x) == label) */
Expression ClassificationAccuracy(const Expression &x, int size, const int *labels);

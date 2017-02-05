#pragma once

#include "Tensor.h"

class Graph;
class Shape;
class Expression {
public:
	/*! Default constructor, constructs an invalid expression handle. */
	Expression();

	/*! Get the graph object this expression belongs to */
	Graph *GetGraph() const { return graph_; }

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

/*! Cross entropy loss: y = -log(x_k) */
Expression CrossEntropy(const Expression &x, int size, const int *labels);

/*! Classification accuracy: y = (argmax(x) == label) */
Expression ClassificationAccuracy(const Expression &x, int size, const int *labels);

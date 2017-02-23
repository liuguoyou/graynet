#pragma once

#include "Tensor.h"

/*! @file */

class Graph;
class Shape;
class Expression {
public:
	/*! Default constructor, constructs an invalid expression handle. */
	Expression();

	/*! Test if this object is a valid expression. */
	bool IsValid() const { return graph_ != nullptr; }

	/*! Get the graph object this expression belongs to. */
	Graph *GetGraph() const { return graph_; }

	/*! Get batch size of this expression. */
	int GetBatchSize() const;

	/*! Get shape of this expression. */
	Shape GetShape() const;

	/*! Test if the value of this expression is dense. */
	bool IsDense() const;

	/*! Test if the value of this expression is sparse. */
	bool IsSparse() const;

	/*! Forward compute this expression, return its value. */
	Tensor Forward() const;

	/*! Backward compute this expression, update gradient accumulation in the graph. */
	void Backward() const;

	/*! @private */
	/*! Construct an expression from a graph and node index. */
	Expression(Graph *graph, int index) : graph_(graph), index_(index) {}

	/*! @private */
	/*! Get node index in graph. */
	int GetNodeIndex() const { return index_; }

private:
	Graph *graph_;
	int index_;
};

/* Graph Operators */

/*! \defgroup Input_Operations */
/*! @{ */

/*! Feeds a single dense tensor to the graph.
 * \param graph The graph to hold the expression.
 * \param shape The shape of the data tensor.
 * \param data The raw dense data. The size of the array must match the size of given shape.
 */
Expression Input(Graph *graph, const Shape &shape, const float *data);

/*! Batched version of \ref Input. Feeds a batch of dense tensors to the graph.
 * \param graph The graph to hold the expression.
 * \param batch_size The size of the input batch.
 * \param shape The shape of the data tensor.
 * \param data The raw data array. The batch instances should be laid sequentially in memory.
 */
Expression BatchInput(Graph *graph, int batch_size, const Shape &shape, const float *data);

/*! Feeds a batched sparse vector input to the graph.
 *
 * The input data should be given as a CSR formatted sparse matrix where each row
 * is an input vector.
 *
 * The CSR matrix is represented as three arrays: an array holding all nonzero values,
 * an array holding the start and end position of each row, and an array holding the
 * column index of each nonzero value.
 *
 * For example, consider the following matrix:
 *
 * \code
 * [ .1 .2  0  0  0  0 ]
 * [  0 .3  0 .4  0  0 ]
 * [  0  0 .5 .6 .7  0 ]
 * [  0  0  0  0  0 .8 ]
 * \endcode
 *
 * In CSR format, the three arrays will be:
 *
 * \code
 * values =         [ .1 .2 .3 .4 .5 .6 .7 .8 ]
 * row indices =    [  0     2     4        7  8 ]
 * column indices = [  0  1  1  3  2  3  4  5 ]
 * \endcode
 *
 * \param graph The graph to hold the expression.
 * \param batch_size The size of the input batch.
 * \param shape The shape of the data vector. Must be of rank 1.
 * \param nonzero_count The total number of nonzero values in the input.
 * \param sparse_data Float array holding nonzero values. The length of this array
 * must be equal to `batch_size + 1`.
 * \param batch_indices Start and end position of each batch. The length of this array
 * must be equal to `nonzero_count`.
 * \param indices Data vector index of each nonzero value. The length of this array
 * must be equal to `nonzero_count`.
 */
Expression BatchSparseVectorInput(Graph *graph, int batch_size, const Shape &shape,
	int nonzero_count, const float *sparse_data, const int *batch_indices, const int *indices);

/*! Lookup embeddings from an embedding table.
 * The embeddings must be given as a matrix where each row is an embedding vector.
 * It must have shape `(emb_count, emb_size)`.
 * The input specifies the row number (starting from 0) of the matrix where the embeddings
 * will be picked. Suppose the input shape of the indices is `(a, b, ..., c)`, the output 
 * shape will be `(a, b, ..., c, emb_size)`.
 * \param embeddings The embedding table. Must be of batch size 1 and rank 2.
 * \param shape The shape of the lookup indices.
 * \param indices The data array holding the lookup indices.
 */
Expression Lookup(const Expression &embeddings, const Shape &shape, const int *indices);

/*! Batched version of \ref lookup.
 * \param embeddings The embedding table. Must be of batch size 1 and rank 2.
 * \param batch_size The batch size of the lookup indices.
 * \param shape The shape of the lookup indices.
 * \param indices The data array holding the lookup indices.
 */
Expression BatchLookup(const Expression &embeddings, int batch_size, const Shape &shape,
	const int *indices);

/*! @} */

/*! \defgroup Arithmetic_Operations */
/*! @{ */

/*! Element-wise addition. */
Expression operator+(const Expression &lhs, const Expression &rhs);

/*! Element-wise addition. */
Expression operator+(float lhs, const Expression &rhs);

/*! Element-wise addition. */
Expression operator+(const Expression &lhs, float rhs);

/*! Element-wise subtraction. */
Expression operator-(const Expression &lhs, const Expression &rhs);

/*! Element-wise subtraction. */
Expression operator-(float lhs, const Expression &rhs);

/*! Element-wise subtraction. */
Expression operator-(const Expression &lhs, float rhs);

/*! Element-wise multiplication. */
Expression operator*(const Expression &lhs, const Expression &rhs);

/*! Element-wise multiplication. */
Expression operator*(float lhs, const Expression &rhs);

/*! Element-wise multiplication. */
Expression operator*(const Expression &lhs, float rhs);

/*! Element-wise division. */
Expression operator/(const Expression &lhs, const Expression &rhs);

/*! Element-wise division. */
Expression operator/(float lhs, const Expression &rhs);

/*! Element-wise division. */
Expression operator/(const Expression &lhs, float rhs);

/*! Element-wise negation: \f$ y = -x \f$ */
Expression operator-(const Expression &x);

/*! Element-wise square function: \f$ y = x^2 \f$ */
Expression Square(const Expression &x);

/*! Element-wise cube function: \f$ y = x^3 \f$ */
Expression Cube(const Expression &x);

/*! Element-wise exponential function: \f$ y = e^x \f$ */
Expression Exp(const Expression &x);

/*! Element-wise logarithm function: \f$ y = \ln{x} \f$ */
Expression Log(const Expression &x);

/*! Absolute value function: \f$ y = \left| x \right| \f$ */
Expression Abs(const Expression &x);

/*! Square root function: \f$ y = \sqrt{x} \f$ */
Expression Sqrt(const Expression &x);

/*! Cubic root function: \f$ y = \sqrt[3]{x} \f$ */
Expression Cbrt(const Expression &x);

/*! Sine function: \f$ y = sin(x) \f$ */
Expression Sin(const Expression &x);

/*! Cosine function: \f$ y = cos(x) \f$ */
Expression Cos(const Expression &x);

/*! Tangent function: \f$ y = tan(x) \f$ */
Expression Tan(const Expression &x);

/*! Inverse sine function: \f$ y = sin^{-1}(x) \f$ */
Expression Asin(const Expression &x);

/*! Inverse cosine function: \f$ y = cos^{-1}(x) \f$ */
Expression Acos(const Expression &x);

/*! Inverse tangent function: \f$ y = tan^{-1}(x) \f$ */
Expression Atan(const Expression &x);

/*! Hyperbolic sine function: \f$ y = \frac{e^x - e^{-x}}{2} \f$ */
Expression Sinh(const Expression &x);

/*! Hyperbolic cosine function: \f$ y = \frac{e^x + e^{-x}}{2} \f$ */
Expression Cosh(const Expression &h);

/*! Hyperbolic tangent function: \f$ y = \frac{e^x - e^{-x}}{e^x + e^{-x}} \f$ */
Expression Tanh(const Expression &x);

/*! Inverse hyperbolic sine function: \f$ y = sinh^{-1}(x) = \ln(x+\sqrt{x^2+1}) \f$ */
Expression Asinh(const Expression &x);

/*! Inverse hyperbolic cosine function: \f$ y = cosh^{-1}(x) = \ln(x+\sqrt{x^2-1}) \f$ */
Expression Acosh(const Expression &x);

/*! Inverse hyperbolic sine function: \f$ y = tanh^{-1}(x) = \frac{1}{2}\ln(\frac{1+x}{1-x}) \f$ */
Expression Atanh(const Expression &x);

/*! Sigmoid function: \f$ y = \frac{1}{1+e^{-x}} \f$ */
Expression Sigmoid(const Expression &x);

/*! ReLU function: \f$ y = \max(x, 0) \f$ */
Expression ReLU(const Expression &x);

/*! @} */

/*! \defgroup Linear_Algebra_Operations */
/*! @{ */
/*! Matrix multiplication */
Expression MatMul(const Expression &lhs, const Expression &rhs);

/*! Vector dot (only sparse dot is supported for now) */
Expression Dot(const Expression &lhs, const Expression &rhs);

/*! @} */

/*! \defgroup Neural_Network_Operations */
/*! @{ */

/*! N-D Convolution (actually cross correlation)
 * x: CHW format
 * filter shape: [output_channels, input_channels, d...]
 * x shape: [input_channels, d...]
 */
Expression Convolution(const Expression &x, const Expression &filter, const Shape &strides, const Shape &padding);

/*! N-D Max Pooling */
Expression MaxPooling(const Expression &x, const Shape &filter_shape, const Shape &strides, const Shape &padding);

/*! N-D Average Pooling */
Expression AvgPooling(const Expression &x, const Shape &filter_shape, const Shape &strides, const Shape &padding);

/*! @} */

/*! \defgroup Tensor_Operations */
/*! @{ */

/*! Reshape tensor */
Expression Reshape(const Expression &x, const Shape &shape);

/*! Flatten a multi-dimensional tensor to 1-D. This is equivalent to `Reshape(x, Shape(x.GetShape().GetSize()))`. */
Expression Flatten(const Expression &x);

/*! Reduce one dimension */
Expression ReduceSum(const Expression &x);

/*! Slice input */
Expression Slice(const Expression &x, const Shape &start, const Shape &size);

/*! @} */

/*! \defgroup Loss_Functions */
/*! @{ */

/*! Softmax function over the last dimension: \f$ y = \frac{e^{x_i}}{\sum{e^{x_i}}} \f$ */
Expression Softmax(const Expression &x);

/*! Soft margin (logistic) loss: \f$ y = \ln(1 + e^{-label*x}) \f$
 *
 * \f$ label \f$ must be -1 or 1.
 */
Expression SoftMargin(const Expression &x, const Expression &label);

/*! Binary cross entropy loss: \f$ y = -label*\ln(x) - (1-label)*\ln(1-x) \f$
 *
 * \f$ label \f$ must be 0 or 1.
 */
Expression BinaryCrossEntropy(const Expression &x, const Expression &label);

/*! Binary classification accuracy: \f$ y = ((x > 0.5) = (label > 0.5)) \f$ */
Expression BinaryClassificationAccuracy(const Expression &x, const Expression &label);

/*! Cross entropy loss: \f$ y = -\ln(x_k) \f$ */
Expression CrossEntropy(const Expression &x, int size, const int *labels);

/*! Classification accuracy: \f$ y = (argmax(x) = label) \f$ */
Expression ClassificationAccuracy(const Expression &x, int size, const int *labels);

/*! @} */

#pragma once

#include "Expression.h"
#include "Tensor.h"

enum InitMethod {
	Constant,
	Uniform,
	Normal,
	GlorotUniform,
	GlorotNormal,
	HeUniform,
	HeNormal,
	Bilinear,
};

class Device;
class Node;
class Optimizer;
class GraphPrivate;
/*! Computation graph object */
class Graph {
public:
	Graph(Device *device);
	virtual ~Graph();

	/*! Get device object associated with this computation graph. */
	Device *GetDevice() const;

	/*! Get device type associated with the device of this graph. */
	DeviceType GetDeviceType() const;

	/*! Reset seed used for random number generation. */
	void SetRandomSeed(int seed);

	/*! Create a new parameter tensor with given shape. */
	Expression AddParameter(const Shape &shape, InitMethod init_method = GlorotUniform);

	/*! Create a new parameter tensor from given initial values. */
	Expression AddParameter(const Shape &shape, const float *initial_values);

	/*! Create or retrieve a parameter by name */
	void DefineParameter(Expression *param, const char *name, const Shape &shape, InitMethod init_method = GlorotUniform);

	/* Retrieve a parameter by name */
	Expression GetParameter(const char *name);

	/*! Push parameter scope */
	void PushScope(const char *name);

	/*! Pop parameter namespace */
	void PopScope();
	
	/*! Clear intermediate nodes.
	 * This function will clear all temporary storage. All Expression objects obtained with this
	 * graph will become invalid except for parameter expressions.
	 */
	void Clear();

	/*! Clear accumulated parameter gradients */
	void ClearParameterGradients();

	/*! Clear paramaters. */
	void ClearParameters();

	/*! Checks whether the gradient of given expression w.r.t. all parameters is correct.
	 * This is done by comparing gradient with numerical differentiation.
	 * \param loss The loss expression to check gradient against. Must be a scalar expression.
	 */
	bool CheckGradient(const Expression &loss, bool verbose = false);

	/*! Save graph to `file_name`. */
	void Save(const char* file_name) const;

	/*! Load graph from `file_name`. */
	int Load(const char* file_name);

	/*! \private */
	Expression AddNode(Node *node, const Shape &output_shape, bool sparse_output = false, int batch_size = -1);

	/*! \private */
	int GetIncrementRandomSeed();

private:
	friend class Expression;
	friend class Optimizer;

	/*! Init parameter with constant */
	void ConstantInit(float *data, int size, float value);

	/*! Init parameter with uniform distribution */
	void UniformInit(float *data, int size, float range);

	/*! Init parameter with normal distribution */
	void NormalInit(float *data, int size, float mean, float variance);

	/*! Append name to scope */
	void AppendScopeName(const char *name);

	/*! Clear forward results without affecting existing nodes */
	void ClearForwardCache();

	/*! Forward propagate an expression */
	Tensor Forward(const Expression &expression);

	/*! Backward propagate an expression*/
	void Backward(const Expression &expression);

	/*! Optimizer update */
	void OptimizerUpdate(Optimizer *optimizer);

	/*! Get shape of node with given index. */
	const Shape &GetNodeShape(int index) const;

	/*! Test if the output of a node is sparse. */
	bool IsNodeOutputSparse(int index) const;

	/*! Get batch size of node with given index. */
	int GetNodeBatchSize(int index) const;

	GraphPrivate *d;
};

#pragma once

/*! This is an internal header file */

#include <initializer_list>
#include <vector>
#include "Graph.h"
#include "Tensor.h"

/*! @private */
class Node {
public:
	/*! Constructor */
	Node(std::initializer_list<int> args) : args_(args) {}

	/*! Destructor */
	virtual ~Node();

	/*! Calculate output shape */
	virtual Shape ForwardShape(const std::vector<Shape> &x_shapes) const;

	/*! Get batch size, only used when GetArguments() == 0. */
	virtual int GetBatchSize() const;

	/*! Do forward computation */
	virtual void Forward(Graph *grpah, const std::vector<const Tensor *> &x, Tensor *y) const;

	/*! Do backward computation */
	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const;

	/*! @private */
	const std::vector<int> &GetArguments() const { return args_; }

private:
	std::vector<int> args_;
};

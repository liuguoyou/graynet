#pragma once

/*! This is an internal header file */

#include <vector>
#include "Expression.h"
#include "Graph.h"
#include "Tensor.h"

#include "initializer_view.h"

/*! \private */
class Node {
public:
	enum NodeFlags {
		NoFlag = 0,
		NoAllocateForwardOutput = 1,
		NoAllocateBackwardOutput = 2,
	};

	/*! Constructor */
	Node() {}

	/*! Constructor */
	Node(std::initializer_list<int> args) : args_(args) {}

	/*! Constructor */
	Node(initializer_view<Expression> args) {
		for (const Expression &expression : args)
			args_.push_back(expression.GetNodeIndex());
	}

	/*! Destructor */
	virtual ~Node();

	/*! Get flags for the node. Returns NoFlag by default. */
	virtual int GetFlags() const;

	/*! Free any associated memory */
	virtual void FreeMemory(Device *device);

	/*! Do forward computation */
	virtual void Forward(Graph *graph, const std::vector<const Tensor *> &x, Tensor *y) const = 0;

	/*! Do backward computation */
	virtual void Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
		const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const = 0;

	/*! \private */
	const std::vector<int> &GetArguments() const { return args_; }

private:
	std::vector<int> args_;
};

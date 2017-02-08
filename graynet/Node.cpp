#include "Node.h"

#include <cstdlib>

Node::~Node() {
}

Shape Node::ForwardShape(const std::vector<Shape> &x_shapes) const {
	abort();
}

int Node::GetBatchSize() const {
	abort();
}

void Node::Forward(Graph *grpah, const std::vector<const Tensor *> &x, Tensor *y) const {
	abort();
}

void Node::Backward(Graph *graph, const std::vector<const Tensor *> &x, const Tensor *y,
	const Tensor *dEdY, const std::vector<Tensor *> &dEdX) const {
	abort();
}

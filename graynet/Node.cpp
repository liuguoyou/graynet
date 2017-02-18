#include "Node.h"
#include "Utils.h"

#include <cstdlib>

Node::~Node() {
}

int Node::GetBatchSize() const {
	DEBUG_BREAK();
	abort();
}

int Node::GetFlags() const {
	return 0;
}

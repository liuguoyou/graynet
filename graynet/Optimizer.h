#pragma once

#include <vector>

class Graph;
class Tensor;
class Optimizer {
public:
	Optimizer(Graph *graph);

	/*! Get the graph object associated with this optimizer */
	Graph *GetGraph() const { return graph_; }

	/*! Update parameters */
	void Update();

protected:
	virtual void UpdateCallback(const std::vector<Tensor> &parameters, const std::vector<Tensor> &gradients) const = 0;
	friend class Graph;

private:
	Graph *graph_;
};

class SGDOptimizer: public Optimizer {
public:
	SGDOptimizer(Graph *graph, float learning_rate);

protected:
	virtual void UpdateCallback(const std::vector<Tensor> &parameters, const std::vector<Tensor> &gradients) const override;

private:
	float learning_rate_;
};

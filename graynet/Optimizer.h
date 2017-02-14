#pragma once

// TODO: Avoid include the vector header since this is public header
#include <vector>

class Graph;
class Tensor;
class OptimizerPrivate;
class Optimizer {
public:
	Optimizer(Graph *graph);

	virtual ~Optimizer();

	/*! Get the graph object associated with this optimizer */
	Graph *GetGraph() const;

	/*! Update parameters */
	void Update();

protected:
	/*! Get the extra number of floats needed for every parameter, defaults to zero. */
	virtual int GetExtraDataCount() const;

	/*! Update parameters */
	virtual void UpdateCallback(const std::vector<Tensor> &parameters,
		const std::vector<Tensor> &gradients,
		const std::vector<Tensor> &extras) const = 0;

	/*! Update parameters, callback for Graph */
	void UpdateCallback(const std::vector<Tensor> &parameters,
		const std::vector<Tensor> &gradients);
	friend class Graph;

private:
	OptimizerPrivate *d;
};

class SGDOptimizer: public Optimizer {
public:
	SGDOptimizer(Graph *graph, float learning_rate);

protected:
	virtual void UpdateCallback(const std::vector<Tensor> &parameters,
		const std::vector<Tensor> &gradients,
		const std::vector<Tensor> &extras) const override;

private:
	float learning_rate_;
};

class AdaGradOptimizer : public Optimizer {
public:
	AdaGradOptimizer(Graph *graph, float initial_learning_rate = 0.01f, float epsilon = 1e-6f);

protected:
	virtual int GetExtraDataCount() const override;
	virtual void UpdateCallback(const std::vector<Tensor> &parameters,
		const std::vector<Tensor> &gradients,
		const std::vector<Tensor> &extras) const override;

private:
	float initial_learning_rate_;
	float epsilon_;
};

class RmsPropOptimizer : public Optimizer {
public:
	RmsPropOptimizer(Graph *grpah, float initial_learning_rate = 0.001f,
		float alpha = 0.9f,
		float epsilon = 1e-6f);

protected:
	virtual int GetExtraDataCount() const override;
	virtual void UpdateCallback(const std::vector<Tensor> &parameters,
		const std::vector<Tensor> &gradients,
		const std::vector<Tensor> &extras) const override;

private:
	float initial_learning_rate_;
	float alpha_;
	float epsilon_;
};
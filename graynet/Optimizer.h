#pragma once

// TODO: Avoid include the vector header since this is public header
#include <vector>

class Graph;
class Tensor;
class OptimizerPrivate;

/*! \defgroup Optimizers */
/*! @{ */

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

/*! Basic stochastic gradient descent optimizer.
 * This optimizer uses a constant learning rate for every update, no momentum/learning rate decay
 * is supported.
 *
 * Update formula: \f[ w_{t+1} \leftarrow w_t - \eta\nabla{w_t} \f]
 */
class SGDOptimizer: public Optimizer {
public:
	/*! Initialize a SGDOptimizer object.
	 * \param learning_rate The constant learning rate \f$ \eta \f$ used for optimizer updates.
	 */
	SGDOptimizer(Graph *graph, float learning_rate);

	/*! Update learning rate to a given value.
	 * This should be called before \ref Optimizer::Update() to affect subsequent optimizer updates.
	 * This function can be used to implement custom learning rate schedule.
	 * \param learning_rate The learning rate to be set for subsequent optimizer updates.
	 */
	void UpdateLearningRate(float learning_rate);

protected:
	virtual void UpdateCallback(const std::vector<Tensor> &parameters,
		const std::vector<Tensor> &gradients,
		const std::vector<Tensor> &extras) const override;

private:
	float learning_rate_;
};

/*! Adaptive gradient optimizer (AdaGrad).
 *
 * Update formula: \f[
 *  g_{t+1} \leftarrow g_t + {\nabla{w_t}}^2
 * \f] \f[
 *  w_{t+1} \leftarrow w_t - \frac{\eta}{\sqrt{g_t+\epsilon}}\nabla{w_t}
 * \f]
 */
class AdaGradOptimizer : public Optimizer {
public:
	/*! Initialize an AdaGradOptimizer object.
	 * \param initial_learning_rate Specify the \f$ \eta \f$ parameter.
	 * \param epsilon Specify the \f$ \epsilon \f$ parameter.
	 */
	AdaGradOptimizer(Graph *graph, float initial_learning_rate = 0.01f, float epsilon = 1e-8f);

protected:
	virtual int GetExtraDataCount() const override;
	virtual void UpdateCallback(const std::vector<Tensor> &parameters,
		const std::vector<Tensor> &gradients,
		const std::vector<Tensor> &extras) const override;

private:
	float initial_learning_rate_;
	float epsilon_;
};

/*! RmsProp optimizer
 *
 * Update formula: \f[
 *  g_{t+1} \leftarrow \alpha g_t + (1-\alpha){\nabla{w_t}}^2
 * \f] \f[
 *  w_{t+1} \leftarrow w_t - \frac{\eta}{\sqrt{g_t+\epsilon}}\nabla{w_t}
 * \f]
 */
class RmsPropOptimizer : public Optimizer {
public:
	/*! Initialize an RmsPropOptimizer object.
	 * \param initializer_learning_rate Specify the \f$ \eta \f$ parameter.
	 * \param alpha Specify the \f$ \alpha \f$ parameter.
	 * \param epsilon Specify the \f$ \epsilon \f$ parameter.
	 */
	RmsPropOptimizer(Graph *grpah, float initial_learning_rate = 0.001f,
		float alpha = 0.9f,
		float epsilon = 1e-8f);

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

/*! @} */

#pragma once

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
	virtual void UpdateCallback(int count,
		Tensor *parameters, Tensor *gradients, Tensor *extras) const = 0;

	/*! Update parameters, callback for Graph */
	void UpdateCallback(int count, Tensor *parameters, Tensor *gradients);
	friend class Graph;

private:
	OptimizerPrivate *d;
};

/*! Basic stochastic gradient descent optimizer.
 * This optimizer uses a constant learning rate for every update, no momentum/learning rate decay
 * is provided. You can use \ref UpdateLearningRate() to implement custom learning
 * rate schedule.
 *
 * Update formula: \f[ \theta_{t+1} \leftarrow \theta_t - \eta\nabla{\theta_t} \f]
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
	virtual void UpdateCallback(int count,
		Tensor *parameters, Tensor *gradients, Tensor *extras) const override;

private:
	float learning_rate_;
};

/*! Adaptive gradient (AdaGrad) optimizer.
 *
 * Update formula: \f[
 *  g_{t+1} \leftarrow g_t + {\nabla{\theta_t}}^2
 * \f] \f[
 *  \theta_{t+1} \leftarrow \theta_t - \frac{\eta}{\sqrt{g_t+\epsilon}}\nabla{\theta_t}
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
	virtual void UpdateCallback(int count,
		Tensor *parameters, Tensor *gradients, Tensor *extras) const override;

private:
	float initial_learning_rate_;
	float epsilon_;
};

/*! RmsProp optimizer.
 *
 * Update formula: \f[
 *  g_{t+1} \leftarrow \alpha g_t + (1-\alpha){\nabla{\theta_t}}^2
 * \f] \f[
 *  \theta_{t+1} \leftarrow \theta_t - \frac{\eta}{\sqrt{g_t+\epsilon}}\nabla{\theta_t}
 * \f]
 */
class RmsPropOptimizer : public Optimizer {
public:
	/*! Initialize an RmsPropOptimizer object.
	 * \param initial_learning_rate Specify the \f$ \eta \f$ parameter.
	 * \param alpha Specify the \f$ \alpha \f$ parameter.
	 * \param epsilon Specify the \f$ \epsilon \f$ parameter.
	 */
	RmsPropOptimizer(Graph *grpah, float initial_learning_rate = 0.001f,
		float alpha = 0.9f,
		float epsilon = 1e-8f);

protected:
	virtual int GetExtraDataCount() const override;
	virtual void UpdateCallback(int count,
		Tensor *parameters, Tensor *gradients, Tensor *extras) const override;

private:
	float initial_learning_rate_;
	float alpha_;
	float epsilon_;
};

/*! Adaptive moment estimation (Adam) optimizer.
 *
 * Update formula: \f[
 *  m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1)\nabla{\theta_t}
 * \f] \f[
 *  v_t \leftarrow \beta_2 v_{t-1} + (1-\beta_2){\nabla{\theta_t}}^2
 * \f] \f[
 *  \hat{m}_t \leftarrow \frac{m_t}{1 - {\beta_1}^t}
 * \f] \f[
 *  \hat{v}_t \leftarrow \frac{v_t}{1 - {\beta_2}^t}
 * \f] \f[
 *  \theta_{t+1} \leftarrow \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t
 * \f]
 */
class AdamOptimizer : public Optimizer {
public:
	/*! Initialize an AdamOptimizer object.
	 * \param initial_learning_rate Specify the \f$ \eta \f$ parameter.
	 * \param beta1 Specify the \f$ \beta_1 \f$ parameter.
	 * \param beta2 Specify the \f$ \beta_2 \f$ parameter.
	 * \param epsilon Specify the \f$ \epsilon \f$ parameter.
	 */
	AdamOptimizer(Graph *graph, float initial_learning_rate = 0.001f,
		float beta1 = 0.9f,
		float beta2 = 0.999f,
		float epsilon = 1e-8f);

protected:
	virtual int GetExtraDataCount() const override;
	virtual void UpdateCallback(int count,
		Tensor *parameters, Tensor *gradients, Tensor *extras) const override;

private:
	float initial_learning_rate_;
	float beta1_, beta2_;
	mutable float beta1_t_, beta2_t_;
	float epsilon_;
};

/*! @} */

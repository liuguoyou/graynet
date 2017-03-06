#include "Graph.h"
#include "Optimizer.h"
#include "Utils.h"

#include <cblas.h>
#include <cmath>
#include <vector>
#ifdef USE_CUDA
#include <cublas_v2.h>
#include <host_defines.h>
#else
#define __device__
#define __host__
#endif

static const int kThreadsPerBlock = 128;

// Helpers for doing tensor arithmetic
// TODO: Better organize these functions
static void saxpy(Device *device, int count, float alpha, const float *from, float *to) {
#ifdef USE_CUDA
	if (device->GetDeviceType() == GPU) {
		CUBLAS_CALL(cublasSaxpy_v2(device->GetCuBLASHandle(),
			count, &alpha, from, 1, to, 1));
	}
	else
#endif
		cblas_saxpy(count, alpha, from, 1, to, 1);
}

#ifdef USE_CUDA
// TODO: Use CUB for better performance
template<typename TransformFunc>
static __global__ void TransformKernel2(int N, const float *from1, const float *from2,
	float *to, TransformFunc func) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		to[i] = func(from1[i], from2[i]);
	}
}
#endif

template<typename TransformFunc>
static void Transform2(Device *device, int N, const float *from1, const float *from2,
	float *to, TransformFunc func) {
#ifdef USE_CUDA
	if (device->GetDeviceType() == GPU) {
		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
		TransformKernel2<<<blocksPerGrid, threadsPerBlock>>>(N, from1, from2, to, func);
	}
	else
#endif
	{
		for (int i = 0; i < N; i++)
			to[i] = func(from1[i], from2[i]);
	}
}

#ifdef USE_CUDA
// TODO: Use CUB for better performance
template<typename TransformFunc>
static __global__ void TransformKernel3(int N, const float *from1, const float *from2,
	const float *from3, float *to, TransformFunc func) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		to[i] = func(from1[i], from2[i], from3[i]);
	}
}
#endif

template<typename TransformFunc>
static void Transform3(Device *device, int N, const float *from1, const float *from2,
	const float *from3, float *to, TransformFunc func) {
#ifdef USE_CUDA
	if (device->GetDeviceType() == GPU) {
		int threadsPerBlock = kThreadsPerBlock;
		int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
		TransformKernel3<<<blocksPerGrid, threadsPerBlock>>>(N, from1, from2, from3, to, func);
	}
	else
#endif
	{
		for (int i = 0; i < N; i++)
			to[i] = func(from1[i], from2[i], from3[i]);
	}
}

class OptimizerPrivate {
public:
	Graph *graph_;

	std::vector<Tensor> extras_;
};

Optimizer::Optimizer(Graph *graph) : d(new OptimizerPrivate()) {
	d->graph_ = graph;
}

Optimizer::~Optimizer() {
	for (const Tensor &extra : d->extras_)
		d->graph_->GetDevice()->FreeMemory(extra.GetData());
	delete d;
}

Graph * Optimizer::GetGraph() const {
	return d->graph_;
}

void Optimizer::Update() {
	d->graph_->OptimizerUpdate(this);
}

int Optimizer::GetExtraDataCount() const {
	return 0;
}

void Optimizer::UpdateCallback(int count, Tensor *parameters, Tensor *gradients) {
	int extras_count = GetExtraDataCount();
	if (extras_count > 0) {
		for (int i = (int)d->extras_.size(); i < count; i++) {
			int batch_size = parameters[i].GetBatchSize();
			const Shape &shape = parameters[i].GetShape();
			int size = extras_count * batch_size * shape.GetSize() * sizeof(float);
			Device *device = d->graph_->GetDevice();

			float *data = (float *)device->AllocateMemory(size);
			device->ZeroMemory(data, size);
			d->extras_.emplace_back(device->GetDeviceType(), batch_size, shape, data);
		}
	}
	UpdateCallback(count, parameters, gradients, d->extras_.data());
}

SGDOptimizer::SGDOptimizer(Graph *graph, float learning_rate)
	: Optimizer(graph), learning_rate_(learning_rate) {
}

void SGDOptimizer::UpdateLearningRate(float learning_rate) {
	learning_rate_ = learning_rate;
}

void SGDOptimizer::UpdateCallback(int count,
	Tensor *parameters, Tensor *gradients, Tensor *extras) const {
	// x -= lr * dEdX
	for (int parameter_id = 0; parameter_id < count; parameter_id++) {
		int size = parameters[parameter_id].GetShape().GetSize();
		float *parameter_data = parameters[parameter_id].GetData();
		float *gradient_data = gradients[parameter_id].GetData();
		if (!gradient_data)
			continue;
		saxpy(GetGraph()->GetDevice(), size, -learning_rate_, gradient_data, parameter_data);
	}
}

AdaGradOptimizer::AdaGradOptimizer(Graph *graph, float initial_learning_rate, float epsilon)
	: Optimizer(graph), initial_learning_rate_(initial_learning_rate), epsilon_(epsilon) {
}

int AdaGradOptimizer::GetExtraDataCount() const {
	return 1;
}

struct AdaGradUpdateG {
	__host__ __device__ float operator()(float g, float grad) {
		return g + grad * grad;
	}
};

struct AdaGradUpdateParam {
	float lr, epsilon;

	__host__ __device__ float operator()(float param, float grad, float g) {
		return param - lr * (grad / (sqrt(g) + epsilon));
	}
};

void AdaGradOptimizer::UpdateCallback(int count,
	Tensor *parameters, Tensor *gradients, Tensor *extras) const {
	for (int parameter_id = 0; parameter_id < count; parameter_id++) {
		int size = parameters[parameter_id].GetShape().GetSize();
		float *parameter_data = parameters[parameter_id].GetData();
		float *gradient_data = gradients[parameter_id].GetData();
		float *g_data = extras[parameter_id].GetData();
		Transform2(GetGraph()->GetDevice(), size, g_data, gradient_data, g_data,
			AdaGradUpdateG());
		Transform3(GetGraph()->GetDevice(), size, parameter_data, gradient_data, g_data,
			parameter_data, AdaGradUpdateParam{ initial_learning_rate_, epsilon_ });
	}
}

RmsPropOptimizer::RmsPropOptimizer(Graph *graph, float initial_learning_rate, float alpha, float epsilon)
	: Optimizer(graph), initial_learning_rate_(initial_learning_rate), alpha_(alpha), epsilon_(epsilon) {
}

int RmsPropOptimizer::GetExtraDataCount() const {
	return 1;
}

struct RmsPropUpdateG {
	float alpha;

	__host__ __device__ float operator()(float g, float grad) {
		return g * alpha + grad * grad * (1 - alpha);
	}
};

void RmsPropOptimizer::UpdateCallback(int count,
	Tensor *parameters, Tensor *gradients, Tensor *extras) const {
	for (int parameter_id = 0; parameter_id < count; parameter_id++) {
		int size = parameters[parameter_id].GetShape().GetSize();
		float *parameter_data = parameters[parameter_id].GetData();
		float *gradient_data = gradients[parameter_id].GetData();
		float *g_data = extras[parameter_id].GetData();
		Transform2(GetGraph()->GetDevice(), size, g_data, gradient_data, g_data,
			RmsPropUpdateG{ alpha_ });
		Transform3(GetGraph()->GetDevice(), size, parameter_data, gradient_data, g_data,
			parameter_data, AdaGradUpdateParam{ initial_learning_rate_, epsilon_ });
	}
}

struct AdamUpdateM {
	float beta1;

	__host__ __device__ float operator()(float m, float grad) {
		return beta1 * m + (1 - beta1) * grad;
	}
};

struct AdamUpdateV {
	float beta2;

	__host__ __device__ float operator()(float v, float grad) {
		return beta2 * v + (1 - beta2) * grad * grad;
	}
};

struct AdamUpdateParam {
	float beta1_t, beta2_t, lr, epsilon;

	__host__ __device__ float operator()(float param, float m, float v) {
		float m_hat = m / (1 - beta1_t);
		float v_hat = v / (1 - beta2_t);
		float x = lr * m_hat / (sqrt(v_hat) + epsilon);
		return param - lr * m_hat / (sqrt(v_hat) + epsilon);
	}
};

AdamOptimizer::AdamOptimizer(Graph *graph, float initial_learning_rate,
	float beta1, float beta2, float epsilon)
	: Optimizer(graph), initial_learning_rate_(initial_learning_rate),
	beta1_(beta1), beta2_(beta2), epsilon_(epsilon), beta1_t_(1.f), beta2_t_(1.f) {
}

int AdamOptimizer::GetExtraDataCount() const {
	return 2;
}

void AdamOptimizer::UpdateCallback(int count,
	Tensor *parameters, Tensor *gradients, Tensor *extras) const {
	beta1_t_ *= beta1_;
	beta2_t_ *= beta2_;
	for (int parameter_id = 0; parameter_id < count; parameter_id++) {
		int size = parameters[parameter_id].GetShape().GetSize();
		float *parameter_data = parameters[parameter_id].GetData();
		float *gradient_data = gradients[parameter_id].GetData();
		float *m_data = extras[parameter_id].GetData();
		float *v_data = m_data + size;
		Transform2(GetGraph()->GetDevice(), size, m_data, gradient_data, m_data,
			AdamUpdateM{ beta1_ });
		Transform2(GetGraph()->GetDevice(), size, v_data, gradient_data, v_data,
			AdamUpdateV{ beta2_ });
		Transform3(GetGraph()->GetDevice(), size, parameter_data, m_data, v_data,
			parameter_data, AdamUpdateParam{ beta1_t_, beta2_t_, initial_learning_rate_, epsilon_ });
	}
}

#include "Graph.h"
#include "Optimizer.h"
#include "Utils.h"

#include <cblas.h>
#ifdef USE_CUDA
#include <cublas_v2.h>
#endif

Optimizer::Optimizer(Graph *graph) : graph_(graph) {}

void Optimizer::Update() {
	graph_->OptimizerUpdate(this);
}

SGDOptimizer::SGDOptimizer(Graph *graph, float learning_rate)
	: Optimizer(graph), learning_rate_(learning_rate) {
}

void SGDOptimizer::UpdateCallback(const std::vector<Tensor> &parameters, const std::vector<Tensor> &gradients) const {
	// x -= lr * dEdX
	for (int parameter_id = 0; parameter_id < (int)parameters.size(); parameter_id++) {
		int size = parameters[parameter_id].GetShape().GetSize();
		float *parameter_data = parameters[parameter_id].GetData();
		float *gradient_data = gradients[parameter_id].GetData();
		if (!gradient_data)
			continue;
		float alpha = -learning_rate_;
#ifdef USE_CUDA
		if (GetGraph()->GetDeviceType() == GPU) {
			CUBLAS_CALL(cublasSaxpy_v2(GetGraph()->GetDevice()->GetCuBLASHandle(),
				size, &alpha, gradient_data, 1, parameter_data, 1));
		}
		else
#endif
			cblas_saxpy(size, alpha, gradient_data, 1, parameter_data, 1);
	}
}

#include "Device.h"
#include "Graph.h"
#include "Node.h"
#include "Optimizer.h"
#include "Utils.h"

#include <cstdio>
#include <ctime>
#include <random>
#include <stack>
#include <vector>
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

/*! @private */
class GraphPrivate {
public:
	Device *device_;

	/*! Computation nodes */
	std::vector<Node *> nodes_;

	/*! Output tensors */
	std::vector<Tensor> outputs_;

	/*! Gradient tensors */
	std::vector<Tensor> gradients_;

	/*! Parameters */
	std::vector<Tensor> parameters_;

	/*! Parameter gradients */
	std::vector<Tensor> parameter_gradients_;

	/*! Scratch spaces used for forward/backward calculations */
	std::vector<Shape> input_shape_scratch_;

	/*! Random generator */
	std::mt19937 rng_;
};

#define PARAMETER_INDEX(index)	(-(index) - 1)

Graph::Graph(Device *device) : d(new GraphPrivate()) {
	d->device_ = device;
	d->rng_ = std::mt19937(clock());
}

Graph::~Graph() {
	delete d;
}

Device *Graph::GetDevice() const {
	return d->device_;
}

DeviceType Graph::GetDeviceType() const {
	return d->device_->GetDeviceType();
}

void Graph::SetRandomSeed(int seed) {
	d->rng_ = std::mt19937(seed);
}

void Graph::ConstantInit(float *data, int size, float value) {
	for (int i = 0; i < size; i++)
		data[i] = value;
}

void Graph::UniformInit(float *data, int size, float range) {
	std::uniform_real_distribution<float> distribution(-range, range);
	for (int i = 0; i < size; i++)
		data[i] = distribution(d->rng_);
}

void Graph::NormalInit(float *data, int size, float mean, float variance) {
	std::normal_distribution<float> distribution(mean, variance);
	for (int i = 0; i < size; i++)
		data[i] = distribution(d->rng_);
}

Expression Graph::AddParameter(const Shape &shape, InitMethod init_method) {
	int size = shape.GetSize();
	float *parameter_data = new float[size];
	switch (init_method) {
	case GlorotUniform: {
		int fan_cnt = 0;
		for (int i = 0; i < shape.GetDimCount(); i++)
			fan_cnt += shape.GetDim(i);
		UniformInit(parameter_data, size, sqrt(6.f / fan_cnt));
		break;
	}

	default:
		abort();
	}
	Expression ret = AddParameter(shape, parameter_data);
	delete parameter_data;
	return ret;
}

Expression Graph::AddParameter(const Shape &shape, const float *initial_values) {
	int size = shape.GetSize();
	float *parameter_data = (float *)d->device_->AllocateMemory(size * sizeof(float), Device::PermanentMemoryPool);
#ifdef USE_CUDA
	if (GetDeviceType() == GPU)
		CUDA_CALL(cudaMemcpy(parameter_data, initial_values, size * sizeof(float), cudaMemcpyHostToDevice));
	else
#endif
		memcpy(parameter_data, initial_values, size * sizeof(float));
	d->parameters_.push_back(Tensor(GetDeviceType(), shape, parameter_data));
	float *parameter_gradient_data = (float *)d->device_->AllocateMemory(size * sizeof(float), Device::PermanentMemoryPool);
	d->device_->ZeroMemory(parameter_gradient_data, size * sizeof(float));
	d->parameter_gradients_.push_back(Tensor(GetDeviceType(), shape, parameter_gradient_data));
	// We use negative indices to represent parameters
	return Expression(this, -(int)d->parameters_.size());
}

void Graph::Clear() {
	// Clear scratch nodes
	for (Node *node : d->nodes_)
		delete node;
	d->nodes_.clear();
	d->outputs_.clear();
	d->gradients_.clear();
	d->device_->ClearMemoryPool(Device::ScratchMemoryPool);
#ifdef USE_CUDA
	d->device_->ClearMemoryPool(Device::PinnedScratchMemoryPool);
#endif
}

void Graph::ClearParameterGradients() {
	for (Tensor parameter_gradient : d->parameter_gradients_)
		d->device_->ZeroMemory(parameter_gradient.GetData(), parameter_gradient.GetShape().GetSize() * sizeof(float));
}

void Graph::ClearForwardCache() {
	for (int i = 0; i < (int)d->nodes_.size(); i++) {
		int batch_size = d->outputs_[i].GetBatchSize();
		Shape shape = d->outputs_[i].GetShape();
		d->outputs_[i] = Tensor(GetDeviceType(), batch_size, shape, nullptr);
		d->gradients_[i] = Tensor(GetDeviceType(), batch_size, shape, nullptr);
	}
	// Input nodes also use scratch tensors so we cannot clean up memory cleanly.
	// As for now this function is only called from CheckGradient() so leaking a bit of
	// memory is not a big issue here.
	//d->device_->ClearMemoryPool(Device::ScratchMemoryPool);
	//d->device_->ClearMemoryPool(Device::PinnedScratchMemoryPool);
}

bool Graph::CheckGradient(const Expression &loss, bool verbose) {
	ClearParameterGradients();

	loss.Forward();
	loss.Backward();

	bool ret = true;
	const float epsilon = 1e-3f;
	const float threshold = 1e-3f;
	for (int parameter_id = 0; parameter_id < (int)d->parameters_.size(); parameter_id++) {
		int size = d->parameters_[parameter_id].GetShape().GetSize();
		Tensor parameter = d->parameters_[parameter_id];
		Tensor gradient = d->parameter_gradients_[parameter_id];
		for (int i = 0; i < size; i++) {
			// Perturb parameter
			float value = parameter.GetValueFlat(i);
			parameter.SetValueFlat(i, value - epsilon);
			ClearForwardCache();
			float y1 = loss.Forward().ReduceSum();

			parameter.SetValueFlat(i, value + epsilon);
			ClearForwardCache();
			float y2 = loss.Forward().ReduceSum();

			parameter.SetValueFlat(i, value);

			// Numerical differentiation
			float num_grad = (y2 - y1) / (epsilon * 2.f);
			float backward_grad = gradient.GetValueFlat(i);
			float diff = fabs(num_grad - backward_grad);
			if (isnan(diff) || diff > threshold) {
				if (verbose) {
					printf("Parameter %d element %d y1: %f y2: %f num: %f backward: %f diff: %f\n",
						parameter_id, i, y1, y2, num_grad, backward_grad, diff);
				}
				ret = false;
			}
		}
	}
	return ret;
}

Expression Graph::AddNode(Node *node) {
	d->nodes_.push_back(node);
	d->input_shape_scratch_.clear();
	for (int input_id : node->GetArguments())
		d->input_shape_scratch_.push_back(GetNodeShape(input_id));
	// Calculate batch size
	int batch_size = 1;
	if (node->GetArguments().empty())
		batch_size = node->GetBatchSize();
	else {
		// Make sure all inputs agree on batch size
		for (int input_id : node->GetArguments()) {
			int cur_batch_size = GetNodeBatchSize(input_id);
			if (cur_batch_size == 1)
				continue;
			if (batch_size == 1)
				batch_size = cur_batch_size;
			else if (batch_size != cur_batch_size)
				abort();
		}
	}
	Shape shape = node->ForwardShape(d->input_shape_scratch_);
	d->outputs_.push_back(Tensor(GetDeviceType(), batch_size, shape, nullptr));
	d->gradients_.push_back(Tensor(GetDeviceType(), batch_size, shape, nullptr));

	int id = (int)d->nodes_.size() - 1;
	return Expression(this, id);
}

Tensor Graph::Forward(const Expression &expression) {
	if (this != expression.GetGraph())
		abort();
	// TODO: Only compute relevant nodes
	std::vector<const Tensor *> x;
	int node_id = expression.GetNodeIndex();
	for (int i = 0; i <= node_id; i++) {
		if (!d->outputs_[i].GetData()) {
			x.clear();
			for (int arg_id : d->nodes_[i]->GetArguments())
				if (arg_id < 0)
					x.push_back(&d->parameters_[PARAMETER_INDEX(arg_id)]);
				else
					x.push_back(&d->outputs_[arg_id]);
			int batch_size = d->outputs_[i].GetBatchSize();
			const Shape &shape = d->outputs_[i].GetShape();
			int size = batch_size * shape.GetSize() * sizeof(float);
			float *output_data = (float*)d->device_->AllocateMemory(size, Device::ScratchMemoryPool);
			d->outputs_[i] = Tensor(GetDeviceType(), batch_size, shape, output_data);
			d->nodes_[i]->Forward(this, x, &d->outputs_[i]);
		}
	}
	if (node_id < 0)
		return d->parameters_[PARAMETER_INDEX(node_id)];
	else
		return d->outputs_[node_id];
}

void Graph::Backward(const Expression &expression) {
	if (this != expression.GetGraph())
		abort();
	int node_id = expression.GetNodeIndex();
	if (node_id < 0)
		return;
	// Expression must be scalar
	const Shape &shape = d->outputs_[node_id].GetShape();
	if (shape.GetSize() != 1)
		abort();
	int batch_size = d->outputs_[node_id].GetBatchSize();
	// Set dE/dE = 1
	float *dEdE_data = (float*)d->device_->AllocateMemory(batch_size * sizeof(float), Device::ScratchMemoryPool);
#ifdef USE_CUDA
	if (d->device_->GetDeviceType() == GPU) {
		float value = 1.f;
		cudnnTensorDescriptor_t tensor_desc;
		CUDNN_CALL(cudnnCreateTensorDescriptor(&tensor_desc));
		CUDNN_CALL(cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, 1));
		CUDNN_CALL(cudnnSetTensor(d->device_->GetCuDNNHandle(), tensor_desc, dEdE_data, &value));
		CUDNN_CALL(cudnnDestroyTensorDescriptor(tensor_desc));
	}
	else
#endif
	{
		for (int i = 0; i < batch_size; i++)
			dEdE_data[i] = 1.f;
	}
	d->gradients_[node_id] = Tensor(GetDeviceType(), batch_size, shape, dEdE_data);
	// Backward propagation
	std::vector<const Tensor *> x;
	std::vector<Tensor *> dEdX;
	std::stack<int, std::vector<int>> stack;
	stack.push(node_id);
	while (!stack.empty()) {
		int i = stack.top();
		stack.pop();
		x.clear();
		dEdX.clear();
		for (int arg_id : d->nodes_[i]->GetArguments())
			if (arg_id < 0) {
				x.push_back(&d->parameters_[PARAMETER_INDEX(arg_id)]);
				dEdX.push_back(&d->parameter_gradients_[PARAMETER_INDEX(arg_id)]);
			}
			else {
				stack.push(arg_id);
				x.push_back(&d->outputs_[arg_id]);
				if (d->gradients_[arg_id].GetData() == nullptr) {
					int batch_size = d->gradients_[arg_id].GetBatchSize();
					const Shape &shape = d->gradients_[arg_id].GetShape();
					int size = shape.GetSize();
					float *data = (float *)d->device_->AllocateMemory(batch_size * size * sizeof(float), Device::ScratchMemoryPool);
					d->device_->ZeroMemory(data, batch_size * size * sizeof(float));
					d->gradients_[arg_id] = Tensor(GetDeviceType(), batch_size, shape, data);
				}
				dEdX.push_back(&d->gradients_[arg_id]);
			}
		const Tensor *y = &d->outputs_[i];
		const Tensor *dEdY = &d->gradients_[i];
		d->nodes_[i]->Backward(this, x, y, dEdY, dEdX);
	}
}

void Graph::OptimizerUpdate(Optimizer *optimizer) {
	optimizer->UpdateCallback(d->parameters_, d->parameter_gradients_);
	Clear();
	ClearParameterGradients();
}

const Shape &Graph::GetNodeShape(int index) const {
	if (index >= 0)
		return d->outputs_[index].GetShape();
	else
		return d->parameters_[PARAMETER_INDEX(index)].GetShape();
}

int Graph::GetNodeBatchSize(int index) const {
	if (index >= 0)
		return d->outputs_[index].GetBatchSize();
	else // Batch size of parameters is always 1.
		return 1;
}

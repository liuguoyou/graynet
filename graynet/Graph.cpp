#include "Device.h"
#include "Graph.h"
#include "Node.h"
#include "Optimizer.h"
#include "Utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <stack>
#include <unordered_map>
#include <vector>
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

static const int kMaxScopeNameLength = 255;

static void FreeTensorMemory(Graph *graph, const Tensor &tensor) {
	if (tensor.IsEmpty())
		return;
	Device *device = graph->GetDevice();
	if (tensor.IsSparse()) {
		device->FreeMemory(tensor.GetSparseData());
		device->FreeMemory(tensor.GetSparseRowIndices());
		device->FreeMemory(tensor.GetSparseColumnIndices());
	}
	else
		device->FreeMemory(tensor.GetData());
}

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

	/*! Parameter name to id map */
	std::unordered_map<std::string, int> parameter_names_; // TODO: Migrate to string_view as key

	/*! Random generator */
	std::mt19937 rng_;

	/*! Ending index of scope names in scope_name_  */
	std::vector<int> scope_indices_;

	/*! Length of current scope name, i.e. the position of null terminator */
	int scope_name_length_;

	/*! Current parameter scope */
	char scope_name_[kMaxScopeNameLength + 1];

	/*! Current random seed */
	int rng_seed_;
};

#define PARAMETER_INDEX(index)	(-(index) - 1)

Graph::Graph(Device *device) : d(new GraphPrivate()) {
	d->device_ = device;
	d->rng_ = std::mt19937(clock());
	d->scope_name_length_ = 0;
	d->scope_name_[0] = 0;
	d->rng_seed_ = 0;
}

Graph::~Graph() {
	// Free scratch memory
	Clear();
	// Free parameter memory
	for (size_t i = 0; i < d->parameters_.size(); i++) {
		FreeTensorMemory(this, d->parameters_[i]);
		FreeTensorMemory(this, d->parameter_gradients_[i]);
	}
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
		for (int i = 0; i < shape.GetRank(); i++)
			fan_cnt += shape.GetDim(i);
		UniformInit(parameter_data, size, sqrt(6.f / fan_cnt));
		break;
	}

	default:
		REPORT_ERROR("Unimplemented initialization method: %d", init_method);
	}
	Expression ret = AddParameter(shape, parameter_data);
	delete[] parameter_data;
	return ret;
}

Expression Graph::AddParameter(const Shape &shape, const float *initial_values) {
	int size = shape.GetSize();
	float *parameter_data = (float *)d->device_->AllocateMemory(size * sizeof(float));
#ifdef USE_CUDA
	if (GetDeviceType() == GPU)
		CUDA_CALL(cudaMemcpy(parameter_data, initial_values, size * sizeof(float), cudaMemcpyHostToDevice));
	else
#endif
		memcpy(parameter_data, initial_values, size * sizeof(float));
	d->parameters_.push_back(Tensor(GetDeviceType(), shape, parameter_data));
	float *parameter_gradient_data = (float *)d->device_->AllocateMemory(size * sizeof(float));
	d->device_->ZeroMemory(parameter_gradient_data, size * sizeof(float));
	d->parameter_gradients_.push_back(Tensor(GetDeviceType(), shape, parameter_gradient_data));
	// We use negative indices to represent parameters
	return Expression(this, -(int)d->parameters_.size());
}

void Graph::AppendScopeName(const char *name) {
	int len = (int)strlen(name);
	int need_dot = (d->scope_name_length_ > 0);
	if (d->scope_name_length_ + need_dot + len >= kMaxScopeNameLength)
		REPORT_ERROR("Parameter scope name too long.");
	if (need_dot)
		d->scope_name_[d->scope_name_length_++] = '.';
	memcpy(d->scope_name_ + d->scope_name_length_, name, len + 1);
	d->scope_name_length_ += len;
}

void Graph::DefineParameter(Expression *param, const char *name, const Shape &shape, InitMethod init_method) {
	if (param->IsValid())
		return;
	int scope_length = d->scope_name_length_;
	AppendScopeName(name);
	std::string str_name(d->scope_name_);
	std::unordered_map<std::string, int>::const_iterator iter = d->parameter_names_.find(str_name);
	if (iter != d->parameter_names_.end()) {
		*param = Expression(this, iter->second);
		if (param->GetShape() != shape) {
			REPORT_ERROR("Requested shape mismatch with registered parameter. "
				"You are probably using the same name for different operations.");
		}
	}
	else {
		*param = AddParameter(shape, init_method);
		d->parameter_names_.emplace(str_name, param->GetNodeIndex());
	}
	d->scope_name_length_ = scope_length;
	d->scope_name_[d->scope_name_length_] = 0;
}

Expression Graph::GetParameter(const char *name) {
	int scope_length = d->scope_name_length_;
	AppendScopeName(name);
	std::string str_name(d->scope_name_);
	std::unordered_map<std::string, int>::const_iterator iter = d->parameter_names_.find(str_name);
	Expression ret;
	if (iter != d->parameter_names_.end())
		ret = Expression(this, iter->second);
	d->scope_name_length_ = scope_length;
	d->scope_name_[d->scope_name_length_] = 0;
	return ret;
}

void Graph::PushScope(const char *name) {
	int len = (int)strlen(name);
	d->scope_indices_.push_back(d->scope_name_length_);
	AppendScopeName(name);
}

void Graph::PopScope() {
	if (d->scope_indices_.empty())
		REPORT_ERROR("Cannot pop scope. It's already at the top level.");
	int index = d->scope_indices_.back();
	d->scope_name_length_ = index;
	d->scope_name_[d->scope_name_length_] = 0;
	d->scope_indices_.pop_back();
}

void Graph::Clear() {
	// Clear scratch nodes and associated memory
	for (size_t i = 0; i < d->nodes_.size(); i++) {
		d->nodes_[i]->FreeMemory(GetDevice());
		FreeTensorMemory(this, d->outputs_[i]);
		FreeTensorMemory(this, d->gradients_[i]);
		delete d->nodes_[i];
	}
	d->nodes_.clear();
	d->outputs_.clear();
	d->gradients_.clear();
}

void Graph::ClearParameterGradients() {
	for (Tensor parameter_gradient : d->parameter_gradients_)
		d->device_->ZeroMemory(parameter_gradient.GetData(), parameter_gradient.GetShape().GetSize() * sizeof(float));
}

void Graph::ClearForwardCache() {
	for (int i = 0; i < (int)d->nodes_.size(); i++) {
		int batch_size = d->outputs_[i].GetBatchSize();
		Shape shape = d->outputs_[i].GetShape();
		FreeTensorMemory(this, d->outputs_[i]);
		FreeTensorMemory(this, d->gradients_[i]);
		d->outputs_[i] = Tensor(GetDeviceType(), batch_size, shape, nullptr);
		d->gradients_[i] = Tensor(GetDeviceType(), batch_size, shape, nullptr);
	}
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
			if (std::isnan(diff) || diff > threshold) {
				if (verbose) {
					printf("Parameter %d element %d x1: %f x2: %f y1: %f y2: %f num: %f backward: %f diff: %f\n",
						parameter_id, i, value - epsilon, value + epsilon, y1, y2, num_grad, backward_grad, diff);
				}
				ret = false;
			}
		}
	}
	return ret;
}

Expression Graph::AddNode(Node *node, const Shape &output_shape, bool sparse_output, int batch_size) {
	d->nodes_.push_back(node);
	// No predefined batch size given, calculate batch size based on input arguments
	if (batch_size == -1) {
		if (node->GetArguments().empty())
			DEBUG_BREAK();
		batch_size = 1;
		// Make sure all inputs agree on batch size
		for (int input_id : node->GetArguments()) {
			int cur_batch_size = GetNodeBatchSize(input_id);
			if (cur_batch_size == 1)
				continue;
			if (batch_size == 1)
				batch_size = cur_batch_size;
			else if (batch_size != cur_batch_size)
				REPORT_ERROR("Batch size mismatch: %d and %d.", batch_size, cur_batch_size);
		}
	}
	d->outputs_.emplace_back(GetDeviceType(), batch_size, output_shape, sparse_output);
	d->gradients_.emplace_back(GetDeviceType(), batch_size, output_shape, sparse_output);

	int id = (int)d->nodes_.size() - 1;
	return Expression(this, id);
}

int Graph::GetIncrementRandomSeed() {
	return d->rng_seed_++;
}

Tensor Graph::Forward(const Expression &expression) {
	// TODO: Only compute relevant nodes
	std::vector<const Tensor *> x;
	int node_id = expression.GetNodeIndex();
	for (int i = 0; i <= node_id; i++) {
		if (d->outputs_[i].IsEmpty()) {
			x.clear();
			for (int arg_id : d->nodes_[i]->GetArguments())
				if (arg_id < 0)
					x.push_back(&d->parameters_[PARAMETER_INDEX(arg_id)]);
				else
					x.push_back(&d->outputs_[arg_id]);
			if (!(d->nodes_[i]->GetFlags() & Node::NoAllocateForwardOutput)) {
				int batch_size = d->outputs_[i].GetBatchSize();
				Shape shape = d->outputs_[i].GetShape();
				int size = batch_size * shape.GetSize() * sizeof(float);
				float *output_data = (float*)d->device_->AllocateMemory(size);
				d->outputs_[i] = Tensor(GetDeviceType(), batch_size, shape, output_data);
			}
			d->nodes_[i]->Forward(this, x, &d->outputs_[i]);
		}
	}
	if (node_id < 0)
		return d->parameters_[PARAMETER_INDEX(node_id)];
	else
		return d->outputs_[node_id];
}

void Graph::Backward(const Expression &expression) {
	int node_id = expression.GetNodeIndex();
	if (node_id < 0)
		return;
	// Expression must be scalar
	const Shape &shape = d->outputs_[node_id].GetShape();
	if (shape.GetSize() != 1)
		REPORT_ERROR("Expression is not a scalar for backward propagation.");
	int batch_size = d->outputs_[node_id].GetBatchSize();
	// Set dE/dE = 1
	float *dEdE_data;
	if (d->gradients_[node_id].IsEmpty()) {
		dEdE_data = (float*)d->device_->AllocateMemory(batch_size * sizeof(float));
		d->gradients_[node_id] = Tensor(GetDeviceType(), batch_size, shape, dEdE_data);
	}
	else
		dEdE_data = d->gradients_[node_id].GetData();
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
	// Back propagation
	// First do a depth first traversal, to determine degree for each node,
	// to omit unnecessary edges for calculating the loss.
	std::vector<int> degrees(d->nodes_.size());
	std::stack<int, std::vector<int>> stack;
	stack.push(node_id);
	while (!stack.empty()) {
		int i = stack.top();
		stack.pop();
		for (int arg_id : d->nodes_[i]->GetArguments())
			if (arg_id >= 0) {
				if (degrees[arg_id] == 0) {
					// FIXME: This is only correct for SparseDotNode
					// For other nodes, we should at least zero the gradient tensor
					// We should probably move ZeroMemory() to tensor and support sparse tensors with ti.
					if (!(d->nodes_[i]->GetFlags() & Node::NoAllocateBackwardOutput)) {
						// Zero gradient for this node
						int batch_size = d->gradients_[arg_id].GetBatchSize();
						Shape shape = d->gradients_[arg_id].GetShape();
						int size = batch_size * shape.GetSize() * sizeof(float);
						float *data = d->gradients_[arg_id].GetData();
						if (data == nullptr) {
							data = (float *)d->device_->AllocateMemory(size);
							d->gradients_[arg_id] = Tensor(GetDeviceType(), batch_size, shape, data);
						}
						d->device_->ZeroMemory(data, size);
					}
					stack.push(arg_id);
				}
				degrees[arg_id]++;
			}
	}
	// Temporary parameter storage for Backward() call
	std::vector<const Tensor *> x;
	std::vector<Tensor *> dEdX;
	// No do backward propagation in topology order
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
				if (--degrees[arg_id] == 0)
					stack.push(arg_id);
				x.push_back(&d->outputs_[arg_id]);
				dEdX.push_back(&d->gradients_[arg_id]);
			}
		const Tensor *y = &d->outputs_[i];
		const Tensor *dEdY = &d->gradients_[i];
		d->nodes_[i]->Backward(this, x, y, dEdY, dEdX);
	}
}

void Graph::OptimizerUpdate(Optimizer *optimizer) {
	optimizer->UpdateCallback((int)d->parameters_.size(),
		d->parameters_.data(), d->parameter_gradients_.data());
	Clear();
	ClearParameterGradients();
}

const Shape &Graph::GetNodeShape(int index) const {
	if (index >= 0)
		return d->outputs_[index].GetShape();
	else
		return d->parameters_[PARAMETER_INDEX(index)].GetShape();
}

bool Graph::IsNodeOutputSparse(int index) const {
	if (index >= 0)
		return d->outputs_[index].IsSparse();
	else
		return d->parameters_[PARAMETER_INDEX(index)].IsSparse();
}

int Graph::GetNodeBatchSize(int index) const {
	if (index >= 0)
		return d->outputs_[index].GetBatchSize();
	else // Batch size of parameters is always 1.
		return 1;
}

// See doc/serialization.rst for detailed serialization format.
static const int kSerializationVersion = 1;

void Graph::Save(const char* file_name) const {
	FILE *out_file = fopen(file_name, "wb");
	if (out_file == nullptr)
		REPORT_ERROR("Open output file failed.");

	// Magic and version number
	int magic = 'YARG';
	fwrite(&magic, 4, 1, out_file);
	fwrite(&kSerializationVersion, 4, 1, out_file);

	// Number of parameters
	int parameter_count = (int)d->parameter_names_.size();
	fwrite(&parameter_count, 4, 1, out_file);

	for (const std::pair<std::string, int> &p : d->parameter_names_) {
		const Tensor &tensor = d->parameters_[PARAMETER_INDEX(p.second)];
		// Parameter name
		int length = (int)p.first.size();
		fwrite(&length, 4, 1, out_file);
		fwrite(p.first.data(), 1, length, out_file);
		// Shape
		if (tensor.GetBatchSize() != 1)
			DEBUG_BREAK();
		const Shape &shape = tensor.GetShape();
		int rank = shape.GetRank();
		fwrite(&rank, 4, 1, out_file);
		for (int i = 0; i < rank; i++) {
			int dim = shape.GetDim(i);
			fwrite(&dim, 4, 1, out_file);
		}
		// Data type
		int data_type = 0; // float
		fwrite(&data_type, 4, 1, out_file);
		// Storage type
		if (tensor.IsSparse())
			DEBUG_BREAK();
		int storage_type = 0; // Dense
		fwrite(&storage_type, 4, 1, out_file);
		// Actual data
		int size = shape.GetSize();
		float *buffer = new float[size];
		tensor.GetValue(buffer);
		fwrite(buffer, 4, size, out_file);
		delete[] buffer;
	}
	fclose(out_file);
}

void Graph::Load(const char *file_name) {
	FILE *in_file = fopen(file_name, "rb");
	if (in_file == nullptr)
		REPORT_ERROR("Open input file failed.");

	// Magic
	int magic;
	fread(&magic, 4, 1, in_file);
	if (magic != 'YARG')
		REPORT_ERROR("File magic mismatch.");
	// Version
	int version;
	fread(&version, 4, 1, in_file);
	if (version != kSerializationVersion)
		REPORT_ERROR("Unsupported serialization version: %d", version);

	// Number of parameters
	int parameter_count;
	fread(&parameter_count, 4, 1, in_file);
	if (parameter_count < 0)
		REPORT_ERROR("Invalid parameter count: %d", parameter_count);

	// Clear existing parameters
	d->parameter_names_.clear();
	for (size_t i = 0; i < d->parameters_.size(); i++) {
		FreeTensorMemory(this, d->parameters_[i]);
		FreeTensorMemory(this, d->parameter_gradients_[i]);
	}
	d->parameters_.clear();
	d->parameter_gradients_.clear();

	for (int i = 0; i < parameter_count; i++) {
		// Parameter name
		std::string name;
		int length;
		fread(&length, 4, 1, in_file);
		if (length <= 0)
			REPORT_ERROR("Invalid parameter name length: %d", length);
		name.resize(length);
		fread(&name[0], 1, length, in_file);
		// Shape
		Shape shape;
		int rank;
		fread(&rank, 4, 1, in_file);
		if (rank <= 0 || rank > kMaxTensorDim)
			REPORT_ERROR("Invalid rank: %d", rank);
		for (int i = 0; i < rank; i++) {
			int dim;
			fread(&dim, 4, 1, in_file);
			if (dim < 1)
				REPORT_ERROR("Invalid dimension %d: %d", i, dim);
			shape.PushDim(dim);
		}
		// Data type
		int data_type;
		fread(&data_type, 4, 1, in_file);
		if (data_type != 0) // Float
			REPORT_ERROR("Unknown data type: %d", data_type);
		// Storage type
		int storage_type;
		fread(&storage_type, 4, 1, in_file);
		if (storage_type != 0) // dense
			REPORT_ERROR("Unknown storage type: %d", storage_type);
		// Data
		int size = shape.GetSize();
		float *data = new float[size];
		fread(data, 4, size, in_file);
		Expression parameter = AddParameter(shape, data);
		d->parameter_names_.emplace(name, parameter.GetNodeIndex());
		delete[] data;
	}
	fclose(in_file);
}

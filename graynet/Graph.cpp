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

void Graph::ClearParameters() {
	d->parameter_names_.clear();
	for (size_t i = 0; i < d->parameters_.size(); i++) {
		FreeTensorMemory(this, d->parameters_[i]);
		FreeTensorMemory(this, d->parameter_gradients_[i]);
	}
	d->parameters_.clear();
	d->parameter_gradients_.clear();
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

const static char* kFileHeader = "GRAY";
const static int kVersion = 0;

void Graph::Save(const char* file_name) const {
	FILE* out_file = fopen(file_name, "wb");
	// store file header and version number
	fwrite(kFileHeader, sizeof(char), std::string(kFileHeader).size(), out_file);
	fwrite(&kVersion, sizeof(int), 1, out_file);

	// store number of paramaters
	int param_num = (int)d->parameter_names_.size();
	fwrite(&param_num, sizeof(int), 1, out_file);

	std::vector<std::string> param_names(d->parameter_names_.size());
	for (const std::pair<const std::string, int>& pr : d->parameter_names_)
		param_names[-pr.second - 1] = pr.first;
	
	// store paramater names
	for (const std::string& name : param_names) {
		int length = (int)name.size();
		fwrite(&length, sizeof(int), 1, out_file);
		fwrite(name.data(), sizeof(char), name.size(), out_file);
	}

	// store paramaters (tensors)
	for (const Tensor& tensor : d->parameters_) {
		char is_sparse = tensor.IsSparse() ? (char)1 : (char)0;
		fwrite(&is_sparse, sizeof(char), 1, out_file);
		int ndim = tensor.GetShape().GetRank();
		fwrite(&ndim, sizeof(int), 1, out_file);
		fwrite(tensor.GetShape().data(), sizeof(int), ndim, out_file);

		// write tensor data type, currently it only can be float. 0 is float, 1 is double.
		int data_type = 0;
		fwrite(&data_type, sizeof(int), 1, out_file);

		int size = tensor.GetShape().GetSize();
		if (is_sparse)
			REPORT_ERROR("Graph serialization only support dense tensor currently.");
		else {
#ifdef USE_CUDA
			if (GetDeviceType() == GPU) {
				float *buffer = new float[size];
				CUDA_CALL(cudaMemcpy(buffer, tensor.GetData(), size * sizeof(float), cudaMemcpyDeviceToHost));
				fwrite(buffer, sizeof(float), size, out_file);
				delete[] buffer;
			}
			else
#endif
			fwrite(tensor.GetData(), sizeof(float), size, out_file);
		}
	}
	
	fclose(out_file);
}

int Graph::Load(const char* file_name) {
	FILE* in_file = fopen(file_name, "rb");

	char file_header[5] = { '\0' };
	int version;
	// load file header and version number
	fread(file_header, sizeof(char), std::string(kFileHeader).size(), in_file);
	fread(&version, sizeof(int), 1, in_file);
	if (strcmp(file_header, kFileHeader) != 0 || version != kVersion) {
		REPORT_ERROR("Graph file header not match or version not match.");
		return 0;
	}

	ClearParameters();

	// load number of paramaters
	int param_num = -1;
	fread(&param_num, sizeof(int), 1, in_file);

	// load paramater names
	std::vector<std::string> param_names(param_num);
	for (int i = 0; i < param_num; ++i) {
		int length = -1;
		fread(&length, sizeof(int), 1, in_file);
		param_names[i].resize(length, '\0');
		fread(&param_names[i][0], sizeof(char), length, in_file);
	}

	// load paramaters (tensors)
	for (int i = 0; i < param_num; ++i) {
		char is_sparse = (char)-1;
		fread(&is_sparse, sizeof(char), 1, in_file);
		int ndim = -1;
		fread(&ndim, sizeof(int), 1, in_file);
		std::vector<int> dims(ndim);
		fread(&dims[0], sizeof(int), ndim, in_file);
		Shape shape;
		for (int d = 0; d < ndim; ++d)
			shape.PushDim(dims[d]);

		// load tensor data type, currently it only can be float. 0 is float, 1 is double.
		int data_type = 0;
		fread(&data_type, sizeof(int), 1, in_file);

		int size = shape.GetSize();
		float *parameter_data = new float[size];
		if (is_sparse) {
			REPORT_ERROR("Graph serialization only support dense tensor currently.");
			return 0;
		}
		else
			fread(parameter_data, sizeof(float), size, in_file);
		
		Expression param = AddParameter(shape, parameter_data);
		delete[] parameter_data;
		d->parameter_names_.emplace(param_names[i], param.GetNodeIndex());
	}

	fclose(in_file);
	return 1;
}

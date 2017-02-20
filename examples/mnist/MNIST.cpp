#include <graynet/graynet.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
static int ToLittleEndian(int x) {
	return _byteswap_ulong(x);
}
#else
static int ToLittleEndian(int x) {
	return __builtin_bswap32(x);
}
#endif

const int kWidth = 28;
const int kHeight = 28;

struct DataPoint {
	char data[kHeight * kWidth];
	int label;
};

static std::vector<DataPoint> LoadMNIST(const char *image_filename, const char *label_filename) {
	std::vector<DataPoint> ret;
	// Load images
	FILE *f = fopen(image_filename, "rb");
	int magic;
	fread(&magic, 4, 1, f);
	magic = ToLittleEndian(magic);
	if (magic != 2051)
		abort();
	int count;
	fread(&count, 4, 1, f);
	count = ToLittleEndian(count);
	int rows, cols;
	fread(&rows, 4, 1, f);
	fread(&cols, 4, 1, f);
	if (ToLittleEndian(rows) != kHeight || ToLittleEndian(cols) != kWidth)
		abort();
	for (int i = 0; i < count; i++) {
		DataPoint data;
		fread(&data.data, 1, kHeight * kWidth, f);
		ret.push_back(data);
	}

	fclose(f);

	f = fopen(label_filename, "rb");
	fread(&magic, 4, 1, f);
	magic = ToLittleEndian(magic);
	if (magic != 2049)
		abort();
	fread(&count, 4, 1, f);
	count = ToLittleEndian(count);
	for (int i = 0; i < count; i++) {
		char label;
		fread(&label, 1, 1, f);
		ret[i].label = label;
	}
	fclose(f);
	return ret;
}

static Expression Model(Expression t) {
#if 1
	t = ReLU(LinearLayer("l1", t, 128));
	t = ReLU(LinearLayer("l2", t, 64));
	t = Softmax(LinearLayer("l3", t, 10));
#else
	t = Reshape(t, Shape(1, kHeight, kWidth));
	t = ReLU(ConvolutionLayer("conv1", t, 32, Shape(3, 3), Shape(1, 1), Shape(0, 0)));
	t = ReLU(ConvolutionLayer("conv2", t, 16, Shape(3, 3), Shape(1, 1), Shape(0, 0)));
	t = MaxPooling(t, Shape(3, 3), Shape(1, 1), Shape(0, 0));
	t = Flatten(t);
	t = ReLU(LinearLayer("l1", t, 128));
	t = Softmax(LinearLayer("l2", t, 10));
#endif
	return t;
}

static void ExtractBatch(std::vector<float> &input_data, std::vector<int> &input_label,
	const std::vector<DataPoint> &dataset, int i, int batch_size) {
	for (int j = 0; j < batch_size; j++) {
		for (int k = 0; k < kHeight * kWidth; k++)
			input_data.push_back(dataset[i + j].data[k] / 256.f);
		input_label.push_back(dataset[i + j].label);
	}
}

int main() {
	Device device(GPU);
	Graph graph(&device);
	graph.SetRandomSeed(0);

	std::vector<DataPoint> trainset = LoadMNIST("D:/Workspace/dataset/train-images.idx3-ubyte", "D:/Workspace/dataset/train-labels.idx1-ubyte");
	std::vector<DataPoint> testset = LoadMNIST("D:/Workspace/dataset/t10k-images.idx3-ubyte", "D:/Workspace/dataset/t10k-labels.idx1-ubyte");
	std::cout << "Trainset size: " << trainset.size() << std::endl;
	std::cout << "Testset size: " << testset.size() << std::endl;

	//SGDOptimizer optimizer(&graph, 0.01f);
	//AdaGradOptimizer optimizer(&graph);
	//RmsPropOptimizer optimizer(&graph);
	AdamOptimizer optimizer(&graph);

	int batch_size = 20;
	float loss = 0, accuracy = 0;

	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < trainset.size() * 100; i += batch_size) {
		int batch_start = i % trainset.size();
		std::vector<float> input_data;
		std::vector<int> input_label;
		ExtractBatch(input_data, input_label, trainset, batch_start, batch_size);
		Expression t = BatchInput(&graph, batch_size, Shape(kHeight * kWidth), input_data.data());
		t = Model(t);
		accuracy += ClassificationAccuracy(t, batch_size, input_label.data()).Forward().ReduceSum();
		t = CrossEntropy(t, batch_size, input_label.data());
		loss += t.Forward().ReduceSum();
		t.Forward();
		t.Backward();
		optimizer.Update();
		if (i % trainset.size() == 0) {
			// Evaluate test set
			float test_accuracy = 0;
			for (int j = 0; j < testset.size(); j += batch_size) {
				std::vector<float> input_data;
				std::vector<int> input_label;
				ExtractBatch(input_data, input_label, testset, j, batch_size);
				Expression t = BatchInput(&graph, batch_size, Shape(kHeight * kWidth), input_data.data());
				t = Model(t);
				test_accuracy += ClassificationAccuracy(t, batch_size, input_label.data()).Forward().ReduceSum();
				graph.Clear();
			}
			std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
			std::cout << i << " Loss: " << loss / trainset.size() << " Accuracy: " << accuracy / trainset.size()
				<< " Test Accuracy: " << test_accuracy / testset.size()
				<< " Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
				<< "ms" << std::endl;
			loss = accuracy = 0;
			start_time = end_time;
		}
	}
	return 0;
}

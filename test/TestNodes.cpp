#include <gtest/gtest.h>
#include <graynet/graynet.h>
#include <random>

class NodeTest: public testing::Test {
public:
	NodeTest() : device(GPU), graph(&device) {
	}

	void CheckGradient(const Expression &loss) {
		EXPECT_TRUE(graph.CheckGradient(loss, true));
	}

	void CheckValue(const Expression &value, const float *expected) {
		const float kAbsErrorAllowance = 1e-4f;
		Tensor result = value.Forward();
		int size = result.GetBatchSize() * result.GetShape().GetSize();
		float *actual = new float[size];
		result.GetValue(actual);
		for (int i = 0; i < size; i++) {
			EXPECT_NEAR(actual[i], expected[i], kAbsErrorAllowance) << "Index is " << i;
		}
		delete actual;
	}

	float *GenerateTestData(const Shape &shape) {
		int count = shape.GetSize();
		float *x = new float[count];
		std::uniform_real_distribution<float> dist(0.f, 1.f);
		for (int i = 0; i < count; i++)
			x[i] = dist(gen);
		return x;
	}

	Device device;
	Graph graph;
	std::mt19937 gen{ 0 };
};

TEST_F(NodeTest, SimpleBatchTest) {
	const float weight_data[] = {
		0.1f, 0.2f, 0.3f,
		0.4f, -0.5f, 0.2f,
	};
	const float x_data[] = {
		0.2f, -0.4f, 0.5f,
		-0.1f, 0.7f, 0.9f,
		0.6f, -0.2f, -0.1f
	};
	Expression weight = graph.AddParameter(Shape(2, 3), weight_data);
	Expression x = BatchInput(&graph, 3, Shape(3), x_data);
	x = MatMul(weight, x);
	const float matvecmul_expected[] = {
		0.09f, 0.38f,
		0.4f, -0.21f,
		-0.01f, 0.32f,
	};
	CheckValue(x, matvecmul_expected);
	x = Softmax(x);
	const int label_data[] = { 1, 1, 0 };
	x = CrossEntropy(x, 3, label_data);
	CheckGradient(x);
}

TEST_F(NodeTest, Input) {
	const float data[] = { -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f };
	Expression x = Input(&graph, Shape(7), data);
	CheckValue(x, data);
}

TEST_F(NodeTest, BatchInput) {
	const float data[] = {
		-1.f, 2.f, 4.f,
		3.f, 5.f, -2.f,
		2.f, -4.f, -6.f,
	};
	Expression x = Input(&graph, Shape(3, 3), data);
	CheckValue(x, data);
}

TEST_F(NodeTest, Lookup) {
	const float embeddings_data[] = {
		0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
		-0.1f, -0.2f, -0.3f, -0.4f, -0.5f,
		0.3f, 0.2f, 0.7f, 1.3f, 0.9f,
		-0.4f, 0.6f, 0.8f, -2.3f, -1.7f,
	};
	const int indices_data[] = {
		3, 0, 0, 1, 2, 1,
	};
	Expression embeddings = graph.AddParameter(Shape(4, 5), embeddings_data);
	Expression x = Lookup(embeddings, Shape(6), indices_data);
	const float expected[] = {
		-0.4f, 0.6f, 0.8f, -2.3f, -1.7f,
		0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
		0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
		-0.1f, -0.2f, -0.3f, -0.4f, -0.5f,
		0.3f, 0.2f, 0.7f, 1.3f, 0.9f,
		-0.1f, -0.2f, -0.3f, -0.4f, -0.5f,
	};
	CheckValue(x, expected);
	x = ReduceSum(x);
	CheckGradient(x);
}

TEST_F(NodeTest, SoftMargin) {
	const float x_data[] = { 0.1f, 0.5f, 0.3f, 0.7f, 0.2f, 0.9f };
	const float label_data[] = { -1.f, -1.f, -1.f, 1.f, 1.f, 1.f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	Expression label = graph.AddParameter(Shape(6), label_data);
	x = SoftMargin(x, label);
	const float expected[] = {
		0.74439666f, 0.97407698f, 0.85435524f, 0.40318605f, 0.59813887f, 0.34115387f
	};
	CheckValue(x, expected);
	x = Dot(x, x);
	CheckGradient(x);
}

TEST_F(NodeTest, BinaryCrossEntropy) {
	const float x_data[] = { 0.1f, 0.5f, 0.3f, 0.7f, 0.2f, 0.9f };
	const float label_data[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	Expression label = graph.AddParameter(Shape(6), label_data);
	x = BinaryCrossEntropy(x, label);
	const float expected[] = {
		0.10536051f, 0.69314718f, 0.35667494f, 0.35667494f, 1.60943791f, 0.10536051f,
	};
	CheckValue(x, expected);
	x = Dot(x, x);
	CheckGradient(x);
}

TEST_F(NodeTest, BinaryClassificationAccuracy) {
	const float x_data[] = { 0.1f, 0.3f, 0.7f, 0.2f, 0.9f, 0.4f };
	const float label_data[] = { 0.f, 0.f, 1.f, 1.f, 0.f, 0.f };
	Expression x = BatchInput(&graph, 6, Shape(1), x_data);
	Expression label = BatchInput(&graph, 6, Shape(1), label_data);
	x = BinaryClassificationAccuracy(x, label);
	const float expected[] = { 1.f, 1.f, 1.f, 0.f, 0.f, 1.f };
	CheckValue(x, expected);
}

TEST_F(NodeTest, CrossEntropy) {
	const float prob[] = { 0.2f, 0.1f, 0.1f, 0.5f, 0.1f };
	const int label = 4;
	Expression x = graph.AddParameter(Shape(5), prob);
	x = CrossEntropy(x, 1, &label);
	const float expected[] = { 2.302585f };
	CheckValue(x, expected);
	CheckGradient(x);
}

TEST_F(NodeTest, Softmax) {
	const float logit[] = { 3.5f, 2.1f, 2.5f, -4.6f, 7.0f, 6.3f };
	const int label = 4;
	Expression x = graph.AddParameter(Shape(6), logit);
	x = Softmax(x);
	const float expected[] = {
		1.95408377e-02f, 4.81871126e-03f, 7.18867246e-03f,
		5.93140904e-06f, 6.47103673e-01f, 3.21342174e-01f
	};
	CheckValue(x, expected);
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Add) {
	const float x_data[] = { 1.f, 2.f, 3.f };
	const float y_data[] = { 4.f, 5.f, 6.f };
	Expression x = graph.AddParameter(Shape(3), x_data);
	Expression y = graph.AddParameter(Shape(3), y_data);
	Expression z = x + y;
	const float z_expected[] = { 5.f, 7.f, 9.f };
	CheckValue(z, z_expected);
	z = Softmax(z);
	int label = 1;
	z = CrossEntropy(z, 1, &label);
	CheckGradient(z);
}

TEST_F(NodeTest, LeftScalarMul) {
	const float x_data[] = { 1.f, 2.f, 3.f };
	Expression x = graph.AddParameter(Shape(3), x_data);
	x = 3.f * x;
	const float expected[] = { 3.f, 6.f, 9.f };
	CheckValue(x, expected);
	x = Softmax(x);
	int label = 1;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, RightScalarMul) {
	const float x_data[] = { 1.f, 2.f, 3.f };
	Expression x = graph.AddParameter(Shape(3), x_data);
	x = x * 3.f;
	const float expected[] = { 3.f, 6.f, 9.f };
	CheckValue(x, expected);
	x = Softmax(x);
	int label = 1;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, MatVecMul) {
	const float x_data[] = {
		0.5f, 1.2f, -2.7f,
		-3.4f, 0.1f, -0.2f,
	};
	const float y_data[] = {
		-1.7f,
		4.6f,
		2.9f,
	};
	Expression x = graph.AddParameter(Shape(2, 3), x_data);
	Expression y = graph.AddParameter(Shape(3), y_data);
	Expression z = MatMul(x, y);
	const float expected[] = { -3.16f, 5.66f };
	CheckValue(z, expected);
	z = Softmax(z);
	const int label = 1;
	z = CrossEntropy(z, 1, &label);
	CheckGradient(z);
}

TEST_F(NodeTest, BroadcastAdd) {
	const float x_data[] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
	};
	const float y_data[] = {
		0.3f,
		-0.2f,
	};
	Expression x = graph.AddParameter(Shape(2, 3), x_data);
	Expression y = graph.AddParameter(Shape(2, 1), y_data);
	Expression z = x + y;
	const float z_expected[] = {
		0.4f, 0.5f, 0.6f,
		0.2f, 0.3f, 0.4f,
	};
	CheckValue(z, z_expected);
	const int label = 1;
	z = Reshape(z, Shape(6));
	z = CrossEntropy(Softmax(z), 1, &label);
	CheckGradient(z);

	const float y2_data[] = {
		0.5f,
	};
	Expression y2 = graph.AddParameter(Shape(1, 1), y2_data);
	Expression z2 = x + y2;
	const float z2_expected[] = {
		0.6f, 0.7f, 0.8f,
		0.9f, 1.0f, 1.1f,
	};
	CheckValue(z2, z2_expected);
	z2 = Reshape(z2, Shape(6));
	z2 = CrossEntropy(Softmax(z2), 1, &label);
	CheckGradient(z2);
}

TEST_F(NodeTest, Sigmoid) {
	const float x_data[] = { -10.f, -5.f, -1.f, 0.f, 1.f, 5.f, 10.f };
	Expression x = graph.AddParameter(Shape(7), x_data);
	x = Sigmoid(x);
	const float expected[] = { 0.9999545f, 0.9933071f, 0.7310586f, 0.5f, 0.2689414f, 0.00669285f, 4.53978687e-5f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Tanh) {
	const float x_data[] = { -10.f, -5.f, -1.f, 0.f, 1.f, 5.f, 10.f };
	Expression x = graph.AddParameter(Shape(7), x_data);
	x = Tanh(x);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Reshape) {
	const float x_data[] = {
		1.f, 2.f,
		3.f, 4.f,
	};
	Expression x = Input(&graph, Shape(2, 2), x_data);
	x = Reshape(x, Shape(4));
	x = Softmax(x);
	const int label = 3;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, ReduceSum1D) {
	const float x_data[] = {
		0.1f, 0.2f, 0.3f,
	};
	Expression x = graph.AddParameter(Shape(3), x_data);
	x = ReduceSum(x);
	const float expected[] = { 0.6f };
	CheckValue(x, expected);
	CheckGradient(x);
}

TEST_F(NodeTest, ReduceSumBatched1D) {
	const float x_data[] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
	};
	Expression x = BatchInput(&graph, 2, Shape(3), x_data);
	x = ReduceSum(x);
	const float expected[] = {
		0.6f,
		1.5f,
	};
	CheckValue(x, expected);
	CheckGradient(x);
}

TEST_F(NodeTest, ReduceSumLarge) {
	int count = 1025;
	float *x_data = GenerateTestData(Shape(count));
	float sum = 0;
	for (int i = 0; i < count; i++)
		sum += x_data[i];
	Expression x = Input(&graph, Shape(count), x_data);
	delete x_data;
	x = ReduceSum(x);
	CheckValue(x, &sum);
	CheckGradient(x);
}

TEST_F(NodeTest, ReduceSumAxis) {
	const float x_data[] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
		0.7f, 0.8f, 0.9f,
	};
	Expression x = graph.AddParameter(Shape(3, 3), x_data);
	Expression row = ReduceSum(x, 1);
	const float expected_row[] = {
		0.6f,
		1.5f,
		2.4f,
	};
	CheckValue(row, expected_row);
	Expression column = ReduceSum(x, 0);
	const float expected_column[] = {
		1.2f, 1.5f, 1.8f,
	};
	CheckValue(column, expected_column);
}

TEST_F(NodeTest, Slice1D) {
	const float x_data[] = {
		1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
	};
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Slice(x, Shape(2), Shape(3));
	const float expected[] = {
		3.f, 4.f, 5.f,
	};
	CheckValue(x, expected);
	x = ReduceSum(x);
	CheckGradient(x);
}

TEST_F(NodeTest, Slice2D) {
	const float x_data[] = {
		0.1f, 0.2f, 0.3f, 0.4f,
		-0.1f, -0.2f, -0.3f, -0.4f,
		0.5f, 0.6f, 0.7f, 0.8f,
	};
	Expression x = graph.AddParameter(Shape(3, 4), x_data);
	x = Slice(x, Shape(1, 1), Shape(2, 3));
	const float expected[] = {
		-0.2f, -0.3f, -0.4f,
		0.6f, 0.7f, 0.8f,
	};
	CheckValue(x, expected);
	x = ReduceSum(x);
	CheckGradient(x);
}

TEST_F(NodeTest, ConcatSimple) {
	const float x_data[] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
	};
	const float y_data[] = {
		-0.1f, -0.2f, -0.3f,
		-0.4f, -0.5f, -0.6f,
	};
	Expression x = graph.AddParameter(Shape(2, 3), x_data);
	Expression y = graph.AddParameter(Shape(2, 3), y_data);
	Expression z1 = Concat({ x, y }, 0);
	const float z1_expected[] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
		-0.1f, -0.2f, -0.3f,
		-0.4f, -0.5f, -0.6f,
	};
	CheckValue(z1, z1_expected);
	CheckGradient(ReduceSum(z1));
	Expression z2 = Concat({ x, y }, 1);
	const float z2_expected[] = {
		0.1f, 0.2f, 0.3f, -0.1f, -0.2f, -0.3f,
		0.4f, 0.5f, 0.6f, -0.4f, -0.5f, -0.6f,
	};
	CheckValue(z2, z2_expected);
	CheckGradient(ReduceSum(z2));
}

TEST_F(NodeTest, Concat3) {
	const float x_data[] = {
		0.9f, -0.8f,
		1.0f, 0.1f,
	};
	const float y_data[] = {
		-0.2f, -0.5f,
		0.3f, 0.7f,
	};
	const float z_data[] = {
		-0.3f, -0.7f,
		0.5f, 0.6f
	};
	Expression x = graph.AddParameter(Shape(2, 2), x_data);
	Expression y = graph.AddParameter(Shape(2, 2), y_data);
	Expression z = graph.AddParameter(Shape(2, 2), z_data);
	Expression r1 = Concat({ x, y, z }, 0);
	const float r1_expected[] = {
		0.9f, -0.8f,
		1.0f, 0.1f,
		-0.2f, -0.5f,
		0.3f, 0.7f,
		-0.3f, -0.7f,
		0.5f, 0.6f,
	};
	CheckValue(r1, r1_expected);
	CheckGradient(ReduceSum(r1));
	Expression r2 = Concat({ x, y, z }, 1);
	const float r2_expected[] = {
		0.9f, -0.8f, -0.2f, -0.5f, -0.3f, -0.7f,
		1.0f, 0.1f, 0.3f, 0.7f, 0.5f, 0.6f,
	};
	CheckValue(r2, r2_expected);
	CheckGradient(ReduceSum(r2));
}

TEST_F(NodeTest, ConvolutionSimple) {
	const float x_data[] = {
		0.1f, 0.2f, 0.3f,
		0.4f, 0.5f, 0.6f,
		0.7f, 0.8f, 0.9f,
	};
	const float filter_data[] = {
		-0.1f, 0.2f,
		0.4f, -0.3f,
	};
	Expression filter = graph.AddParameter(Shape(1, 1, 2, 2), filter_data);
	Expression x = graph.AddParameter(Shape(1, 3, 3), x_data);
	x = Convolution(x, filter, Shape(1, 1), Shape(0, 0));
	const float expected[] = {
		0.04f, 0.06f,
		0.1f, 0.12f,
	};
	CheckValue(x, expected);
	x = Reshape(x, Shape(4));
	x = Softmax(x);
	const int label = 1;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, PoolingSimple) {
	const float x_data[] = {
		0.3f, 0.5f, -0.1f,
		0.0f, -0.7f, 0.2f,
		0.9f, 0.1f, 0.3f,
	};
	Expression x = graph.AddParameter(Shape(1, 3, 3), x_data);
	x = MaxPooling(x, Shape(2, 2), Shape(1, 1), Shape(0, 0));
	const float expected[] = {
		0.5f, 0.5f,
		0.9f, 0.3f,
	};
	CheckValue(x, expected);
	x = Reshape(x, Shape(4));
	x = Softmax(x);
	const int label = 1;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, AvgPooling1D) {
	// TODO: Automatic implement 1D in pooling operator.
	const float x_data[] = { 1.f, 2.f, 3.f, 4.f, 5.f };
	Expression x = graph.AddParameter(Shape(1, 5, 1), x_data);
	x = AvgPooling(x, Shape(3, 1), Shape(1, 1), Shape(1, 0));
	const float expected[] = { 1.5f, 2.f, 3.f, 4.f, 4.5f };
	CheckValue(x, expected);
	x = Softmax(Reshape(x, Shape(5)));
	const int label = 1;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, BatchMaxPooling2D) {
	const float x_data[] = {
		0.1f, 0.3f, 0.5f,
		0.2f, -0.6f, 0.7f,
		0.1f, 0.5f, 0.9f,

		-0.1f, -0.3f, 0.5f,
		-0.4f, -0.2f, -0.7f,
		0.3f, -0.9f, -0.5f,
	};
	Expression x = BatchInput(&graph, 2, Shape(1, 3, 3), x_data);
	x = MaxPooling(x, Shape(3, 3), Shape(1, 1), Shape(1, 1));
	const float expected[] = {
		0.3f, 0.7f, 0.7f,
		0.5f, 0.9f, 0.9f,
		0.5f, 0.9f, 0.9f,

		-0.1f, 0.5f, 0.5f,
		0.3f, 0.5f, 0.5f,
		0.3f, 0.3f, -0.2f,
	};
	CheckValue(x, expected);
}

TEST_F(NodeTest, Dot) {
	const float x_data[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
	const float y_data[] = { -0.1f, -0.3f, -0.7f, 0.2f, 0.3f };
	Expression x = graph.AddParameter(Shape(5), x_data);
	Expression y = graph.AddParameter(Shape(5), y_data);
	Expression z = Dot(x, y);
	const float expected[] = { -0.05f };
	CheckValue(z, expected);
	CheckGradient(z);

	Expression z2 = Dot(x, x);
	const float expected2[] = { 0.55f };
	CheckValue(z2, expected2);
	CheckGradient(z2);
}

TEST_F(NodeTest, SparseDot) {
	/*
	 * .1 .2  0  0  0  0
	 *  0 .3  0 .4  0  0
	 *  0  0 .5 .6 .7  0
	 *  0  0  0  0  0 .8
	 */
	const float elems[] = {
		0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f
	};
	const int batch_indices[] = {
		0, 2, 4, 7, 8
	};
	const int indices[] = {
		0, 1, 1, 3, 2, 3, 4, 5
	};
	const float weight_data[] = {
		0.1f, -0.1f, 0.2f, -0.2f, 0.3f, -0.3f
	};
	Expression weight = graph.AddParameter(Shape(6), weight_data);
	Expression x = BatchSparseVectorInput(&graph, 4, Shape(6), 8,
		elems, batch_indices, indices);
	x = Dot(x, weight);
	const float expected[] = {
		-0.01f, -0.11f, 0.19f, -0.24f
	};
	CheckValue(x, expected);
	CheckGradient(x);
}

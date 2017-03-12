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

	void CheckValue(const Expression &value, const float *expected, double abs_error_allowance = 1e-4f) {
		Tensor result = value.Forward();
		int size = result.GetBatchSize() * result.GetShape().GetSize();
		float *actual = new float[size];
		result.GetValue(actual);
		for (int i = 0; i < size; i++) {
			EXPECT_NEAR(actual[i], expected[i], abs_error_allowance) << "Index is " << i;
		}
		delete[] actual;
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

TEST_F(NodeTest, ClassificationAccuracy) {
	const float predict_prob[] = { 
		0.1f, 0.3f, 0.6f, 
		0.2f, 0.5f, 0.3f, 
		0.8f, 0.1f, 0.1f, 
		0.3f, 0.3f, 0.4f,
	};
	const int label_data[] = { 1, 1, 0, 2 };
	const int data_size = 4;
	Expression x = BatchInput(&graph, data_size, Shape(3), predict_prob);
	x = ClassificationAccuracy(x, data_size, label_data);
	const float expected[] = { 0.f, 1.f, 1.f, 1.f };
	CheckValue(x, expected);
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

TEST_F(NodeTest, Square) {
	const float x_data[] = { -3.14f, -1.23f, 0.f, 1.f, 1.23f, 3.14f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Square(x);
	const float expected[] = { 9.85960e+00f, 1.51290e+00f, 0.00000e+00f,
		1.00000e+00f, 1.51290e+00f, 9.85960e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Cube) {
	const float x_data[] = { -1.54f, -1.23f, 0.f, 1.f, 1.23f, 1.54f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Cube(x);
	const float expected[] = { -3.65226e+00f, -1.86087e+00f, 0.00000e+00f, 
		1.00000e+00f, 1.86087e+00f, 3.65226e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Exp) {
	const float x_data[] = { -1.54f, -1.23f, 0.f, 1.f, 1.23f, 1.54f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Exp(x);
	const float expected[] = { 2.14381e-01f, 2.92293e-01f, 1.00000e+00f, 
		2.71828e+00f, 3.42123e+00f, 4.66459e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Log) {
	const float x_data[] = { 0.9f, 1.f, 1.23f, 1.54f , 2.14f, 3.0f};
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Log(x);
	const float expected[] = { -1.05361e-01f, 0.00000e+00f, 2.07014e-01f, 
		4.31782e-01f, 7.60806e-01f, 1.09861e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Abs) {
	const float x_data[] = { -1.54f, -1.23f, 1.f, 1.23f, 1.54f };
	Expression x = graph.AddParameter(Shape(5), x_data);
	x = Abs(x);
	const float expected[] = { 1.54f, 1.23f, 1.f, 1.23f, 1.54f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Sqrt) {
	const float x_data[] = { 0.9f, 1.f, 1.23f, 1.54f , 2.14f, 3.0f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Sqrt(x);
	const float expected[] = { 9.48683e-01f, 1.00000e+00f, 1.10905e+00f, 
		1.24097e+00f, 1.46287e+00f, 1.73205e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Cbrt) {
	const float x_data[] = { -0.9f, 1.f, 1.23f, 1.54f , 2.14f, 3.0f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Cbrt(x);
	const float expected[] = { -9.65489e-01f, 1.00000e+00f, 1.07144e+00f, 
		1.15480e+00f, 1.28866e+00f, 1.44225e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Sin) {
	const float x_data[] = { -14.0f, -1.23f, 0.f, 1.f, 1.23f, 1.54f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Sin(x);
	const float expected[] = { -9.90607e-01f, -9.42489e-01f, 0.00000e+00f, 
		8.41471e-01f, 9.42489e-01f, 9.99526e-01f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Cos) {
	const float x_data[] = { -154.0f, -1.23f, 0.f, 1.f, 1.23f, 1.54f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Cos(x);
	const float expected[] = { -9.98081e-01f, 3.34238e-01f, 1.00000e+00f, 
		5.40302e-01f, 3.34238e-01f, 3.07915e-02f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Tan) {
	const float x_data[] = { -1.25f, -1.23f, 0.f, 1.f, 1.23f, 1.27f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Tan(x);
	const float expected[] = { -3.00957e+00f, -2.81982e+00f, 0.00000e+00f, 
		1.55741e+00f, 2.81982e+00f, 3.22363e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Asin) {
	const float x_data[] = { -0.9f, -0.23f, 0.0f, 0.1f, 0.53f, 0.9f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Asin(x);
	const float expected[] = { -1.11977e+00f, -2.32078e-01f, 0.00000e+00f, 
		1.00167e-01f, 5.58601e-01f, 1.11977e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Acos) {
	const float x_data[] = { -0.9f, -0.23f, 0.0f, 0.1f, 0.53f, 0.9f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Acos(x);
	const float expected[] = { 2.69057e+00f, 1.80287e+00f, 1.57080e+00f, 
		1.47063e+00f, 1.01220e+00f, 4.51027e-01f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Atan) {
	const float x_data[] = { -154.0f, -1.23f, 0.f, 1.f, 1.23f, 1.34f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Atan(x);
	const float expected[] = { -1.56430e+00f, -8.88174e-01f, 0.00000e+00f, 
		7.85398e-01f, 8.88174e-01f, 9.29688e-01f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Sinh) {
	const float x_data[] = { -1.2f, -1.0f, 0.f, 1.f, 1.23f, 1.34f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Sinh(x);
	const float expected[] = { -1.50946e+00f, -1.17520e+00f, 0.00000e+00f, 
		1.17520e+00f, 1.56447e+00f, 1.77860e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Cosh) {
	const float x_data[] = { -1.2f, -1.0f, 0.f, 1.f, 1.23f, 1.34f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Cosh(x);
	const float expected[] = { 1.81066e+00f, 1.54308e+00f, 1.00000e+00f, 
		1.54308e+00f, 1.85676e+00f, 2.04044e+00f };
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
	const float expected[] = { -1.00000e+00f, -9.99909e-01f, -7.61594e-01f, 
		0.00000e+00f, 7.61594e-01f, 9.99909e-01f, 1.00000e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Asinh) {
	const float x_data[] = { -1.50946e+00f, -1.17520e+00f, 0.00000e+00f,
		1.17520e+00f, 1.56447e+00f, 1.77860e+00f  };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Asinh(x);
	const float expected[] = { -1.2f, -1.0f, 0.f, 1.f, 1.23f, 1.34f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Acosh) {
	const float x_data[] = { 1.54308e+00f, 1.64308e+00f, 1.81066e+00f, 
		1.85676e+00f, 2.04044e+00f, 3.04044e+00f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = Acosh(x);
	const float expected[] = { 9.99999e-01f, 1.08072e+00f, 1.20000e+00f, 
		1.23000e+00f, 1.34000e+00f, 1.77694e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Atanh) {
	const float x_data[] = { -0.9f, -0.8f, -0.7f, 0.0f, 0.7f, 0.8f, 0.9f };
	Expression x = graph.AddParameter(Shape(7), x_data);
	x = Atanh(x);
	const float expected[] = { -1.47222e+00f, -1.09861e+00f, -8.67301e-01f, 
		0.00000e+00f, 8.67301e-01f, 1.09861e+00f, 1.47222e+00f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, Sigmoid) {
	const float x_data[] = { -10.f, -5.f, -1.f, 0.f, 1.f, 5.f, 10.f };
	Expression x = graph.AddParameter(Shape(7), x_data);
	x = Sigmoid(x);
	const float expected[] = { 4.53979e-05f, 6.69285e-03f, 2.68941e-01f, 5.00000e-01f, 
		7.31059e-01f, 9.93307e-01f, 9.99955e-01f };
	CheckValue(x, expected);
	x = Softmax(x);
	const int label = 0;
	x = CrossEntropy(x, 1, &label);
	CheckGradient(x);
}

TEST_F(NodeTest, ReLU) {
	const float x_data[] = { -0.9f, -0.8f, -0.7f, 0.7f, 0.8f, 0.9f };
	Expression x = graph.AddParameter(Shape(6), x_data);
	x = ReLU(x);
	const float expected[] = { 0.f, 0.f, 0.f, 0.7f, 0.8f, 0.9f };
	CheckValue(x, expected);
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
	const double kAbsTolerance = 1e-3;
	int count = 1025;
	float *x_data = GenerateTestData(Shape(count));
	float sum = 0;
	for (int i = 0; i < count; i++)
		sum += x_data[i];
	Expression x = Input(&graph, Shape(count), x_data);
	delete x_data;
	x = ReduceSum(x);
	CheckValue(x, &sum, kAbsTolerance);
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

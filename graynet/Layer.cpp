#include "Graph.h"
#include "Layer.h"
#include "Utils.h"

Expression LinearLayer::operator()(const char *name, const Expression &x, int output_dim) {
	Graph *graph = x.GetGraph();
	graph->PushScope(name);
	const Shape &shape = x.GetShape();
	int input_dim = shape.GetDim(shape.GetRank() - 1);
	graph->DefineParameter(&w, "w", Shape(input_dim, output_dim));
	graph->DefineParameter(&b, "b", Shape(output_dim));
	result = MatMul(x, w) + b;
	graph->PopScope();
	return result;
}

Expression ConvolutionLayer::operator()(const char *name, const Expression &x,
	int output_channels, const Shape &kernel, const Shape &strides, const Shape &padding) {
	Graph *graph = x.GetGraph();
	graph->PushScope(name);
	if (!w.IsValid()) {
		Shape w_shape;
		w_shape.PushDim(output_channels);
		w_shape.PushDim(x.GetShape().GetDim(0));
		w_shape.PushDims(kernel);
		graph->DefineParameter(&w, "w", w_shape);
	}
	if (!b.IsValid()) {
		Shape b_shape;
		b_shape.PushDim(output_channels);
		for (int i = 0; i < kernel.GetRank(); i++)
			b_shape.PushDim(1);
		graph->DefineParameter(&b, "b", b_shape);
	}
	result = Convolution(x, w, strides, padding) + b;
	graph->PopScope();
	return result;
}

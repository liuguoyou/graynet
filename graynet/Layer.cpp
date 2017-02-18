#include "Graph.h"
#include "Layer.h"
#include "Utils.h"

#include <cstdlib>

Expression LinearLayer::operator()(const char *name, const Expression &x, int output_dim) {
	Graph *graph = x.GetGraph();
	graph->PushScope(name);
	const Shape &shape = x.GetShape();
	if (shape.GetDimCount() != 1) // TODO: Relax this constraint
		REPORT_ERROR("Only 1D input is supported.");
	int input_dim = shape.GetDim(0);
	graph->DefineParameter(&w, "w", Shape(output_dim, input_dim));
	graph->DefineParameter(&b, "b", Shape(output_dim));
	result = MatMul(w, x) + b;
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
		for (int i = 0; i < kernel.GetDimCount(); i++)
			b_shape.PushDim(1);
		graph->DefineParameter(&b, "b", b_shape);
	}
	result = Convolution(x, w, strides, padding);// +b; FIXME: TOO SLOW DISABLE FOR NOW
	graph->PopScope();
	return result;
}

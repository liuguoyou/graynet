#pragma once

#include "Expression.h"

template<typename Derived>
struct Layer {
	Expression result;

	operator Expression() { return result; }
};

struct LinearLayer : public Layer<LinearLayer> {
	Expression w, b;

	LinearLayer(const char *name, const Expression &x, int output_dim) {
		operator()(name, x, output_dim);
	}

	Expression operator()(const char *name, const Expression &x, int output_dim);
};

struct ConvolutionLayer : public Layer<ConvolutionLayer> {
	Expression w, b;

	ConvolutionLayer(const char *name, const Expression &x,
		int output_channels, const Shape &kernel, const Shape &strides, const Shape &padding) {
		operator()(name, x, output_channels, kernel, strides, padding);
	}

	Expression operator()(const char *name, const Expression &x,
		int output_channels, const Shape &kernel, const Shape &strides, const Shape &padding);
};

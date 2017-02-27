# graynet

[![Documentation status](https://readthedocs.org/projects/graynet/badge/?version=latest)](http://graynet.readthedocs.io/en/latest/) [![Build Status](https://travis-ci.org/wishstudio/graynet.svg?branch=master)](https://travis-ci.org/wishstudio/graynet)

*efficiency, expressibility, bloatless - choose three*

A new C++ immediate mode deep learning framework.

WIP, check back later.

By using the term *immediate mode* we emphasize an easy to use programming interface.

Dynamic neural network does not necessarily mean immediate mode neural network. As of the time of writing, all other dynamic neural network toolkits (Chainer, DyNet, and PyTorch) still use retain mode. A typical workflow in these toolkits look like:

```
<forward define model parameters>
<loading data>
for each batch in data:
  <define model for this batch using defined model parameters>
  <do forward and backward pass>
```

This approach has two major limitations:

1. Loses the ability to automatically inference tensor shapes, which adds maintenance effort when you are rapidly iterating new network structures/parameters. These toolkits may be able to get over this by developing complementary mechanisms which does model definition and shape inference in model definition stage. But this is essentially writing the *same thing* two times. It bloats code
2. Makes the code harder to read and maintain by requiring you to place some part of model definition beforehand and some part later. If you want to add a new layer, you have to forward define the layer to be able to use it. You also need to make sure the defined shape and the input shape matches.

GrayNet overcome such issues with several unique features:

1. Define and run a neural network with completely immediate mode programming. Model definition can be defined self-contained in a function. This allows very expressive definitions for a model:

 ```C++
// This is a self-contained model definition for MNIST.
// No complementary forward definitions required for using this function.
Expression Model(Expression t) {
  if (mlp) {
    t = ReLU(LinearLayer("l1", t, 128));
    t = ReLU(LinearLayer("l2", t, 64));
    t = Softmax(LinearLayer("l3", t, 10));
  }
  else {
    t = Reshape(t, Shape(1, kHeight, kWidth));
    t = ReLU(ConvolutionLayer("conv1", t, 32, Shape(3, 3), Shape(1, 1), Shape(0, 0)));
    t = ReLU(ConvolutionLayer("conv2", t, 16, Shape(3, 3), Shape(1, 1), Shape(0, 0)));
    t = MaxPooling(t, Shape(3, 3), Shape(1, 1), Shape(0, 0));
    t = Flatten(t);
    t = ReLU(LinearLayer("l1", t, 128));
    t = Softmax(LinearLayer("l2", t, 10));
  }
}
 ```

2. Minimal framework overhead. By using a custom GPU memory allocator and written all code in C++. GrayNet eliminates most framework overhead when you don't need it. But we don't stop here. As a compiled programming language, C++ is often criticized for having slow compilation times, which is bad for rapid iteration. In GrayNet, we do very hard job of optimizing compile time overhead. As an example, the whole repo of GrayNet builds in one minute using VS2015 with a laptop, including all the tests and examples. Unlike many other C++ frameworks, GrayNet avoids exposing bloated STL templates (even without vector!) to your code by carefully design the library interface and data structures. You also do not need to install and configure CUDA toolkit to use GPU if you use the dynamic library redistribution. So you can enjoy the efficiency of C++ without its compilation curses.

# Documentation
The documentation of GrayNet can be retrieved at https://graynet.rtfd.io.

# Examples
Code examples are located at the [examples](https://github.com/wishstudio/graynet/tree/master/examples) directory in the repo.

# License
TBD. Will use a very permissive license.

Operations
==========

.. doxygenclass:: Expression
   :members:

Input Operations
----------------

.. doxygengroup:: Input_Operations
   :members:
   :content-only:

Arithmetic Operations
---------------------

All binary arithmetic operations support NumPy style broadcasting semantics.

When operating on two tensors, GrayNet compares their shapes element-wise and check if they are compatible. Two dimensions are compatible when:

1. The two dimensions have equal size, or
2. One of the dimension has size 1.

If the two dimensions have equal size, no additional processing is needed for that particular dimension.
Otherwise, if one has size 1 and the other has not, the former dimension will be broadcasted to the same size as the latter, by expanding the same data multiple times.

For example, consider the following tensors:

.. code-block:: none

   x = [ 1 2 3 ]  y = [ 1 ]
       [ 4 5 6 ]      [ 2 ]
       [ 7 8 9 ]      [ 3 ]

where `x` has shape `(3, 3)` and `y` has shape `(3, 1)`. When doing `x + y`, the second dimension of `y` will be broadcasted to the same size as of `x`.
Logically, it is like expanding `y` to `(3, 3)`:

.. code-block:: none

   x = [ 1 2 3 ]  y = [ 1 1 1 ]
       [ 4 5 6 ]      [ 2 2 2 ]
       [ 7 8 9 ]      [ 3 3 3 ]

before doing the addition.

Note: In NumPy, arrays do not need to have the same number of dimensions. This is currently not implemented in GrayNet but is planned.

.. doxygengroup:: Arithmetic_Operations
   :members:
   :content-only:

Linear Algebra Operations
-------------------------

.. doxygengroup:: Linear_Algebra_Operations
   :members:
   :content-only:

Neural Network Operations
-------------------------

.. doxygengroup:: Neural_Network_Operations
   :members:
   :content-only:

Tensor Operations
-----------------

.. doxygengroup:: Tensor_Operations
   :members:
   :content-only:

Loss Functions
--------------

.. doxygengroup:: Loss_Functions
   :members:
   :content-only:

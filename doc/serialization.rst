Serialization format
====================

.. code-block:: none

   1. 4 bytes, "GRAY", file format header;
   2. 1 int32, version number;
   3. 1 int32, count, the number of paramaters;
   4. output `count` paramater names:
        for each paramater name:
          A. 1 int32, length, the number of characters of the name;
          B. `length` characters.
   5. output `count` paramater(tensor):
        for each paramater(tensor):
          A. 1 byte, represents it is sparse or not, 0 is dense, 1 is sparse;
          B. Shape object:
             a. 1 int32, ndim_, number of dimension;
             b. `ndim_` int32, store `dims_` array.
          C. 1 int32, data type, 0 is float, 1 is double;
          D. data_ array. length is shape.GetSize(). tensor data type may be float or double.

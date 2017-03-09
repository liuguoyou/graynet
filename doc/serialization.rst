Serialization format
====================

.. code-block:: none

   struct {
     char magic[4]; // GRAY
     int version; // 1
     int parameter_count;
     Parameter parameters[parameter_count];
   };

   struct Parameter {
      int name_len; // Must be positive
      char name[name_len]; // No NULL terminator
      int rank; // Must be positive
      int dims[rank]; // Every dimension must be positive
      int data_type; // 0 - float, 1 - double
      int storage_type; // 0 - dense, 1 - sparse
      DType data[]; // DType specified by data_type, size equals to production of all dims.
   };

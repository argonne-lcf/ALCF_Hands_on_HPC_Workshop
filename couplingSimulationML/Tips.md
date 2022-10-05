Tips for using the Python and Numpy C APIs
- The NumPy C API functions that return PyObject* will return null on failure. To handle Python errors, check for null.
- NumPy supports a lot of array formats. You can't assume the C-style API (contiguous in memory, row major, etc.), so you may want to use the convenience functions PyArray_GETPTR1,... , PyArray_GETPTR4 for indexing into arrays.
- Python errors aren't automatically printed the way they are if you are running Python directly. Some tips for debugging:
    - You may want to run your Python code directly in Python while debugging
    - One way to check if an error occurred and print it is: 
      ```
        if (PyErr_Occurred() != nullptr) {
          PyErr_Print();
        }
      ```

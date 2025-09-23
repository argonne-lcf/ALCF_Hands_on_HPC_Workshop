#include "argparse.hpp"
#include <sycl/sycl.hpp>
#include <vector>

template <class T> class Matrix {
public:
  Matrix(size_t rows, size_t cols)
      : mRows(rows), mCols(cols), mData(rows * cols){};
  T &operator()(size_t i, size_t j) { return mData[i * mCols + j]; };
  T operator()(size_t i, size_t j) const { return mData[i * mCols + j]; };
  T *data() { return mData.data(); };

private:
  size_t mRows;
  size_t mCols;
  std::vector<T> mData;
};

// Inspired by Codeplay compute cpp hello-world
int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("3_buffer_2d");

  program.add_argument("-g","--global")
   .help("Global Range")
   .default_value(1)
   .action([](const std::string& value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }
  const auto global_range = program.get<int>("-g");

  //  _       _   _
  // |_)    _|_ _|_ _  ._
  // |_) |_| |   | (/_ |
  //

  Matrix<int> A(global_range, global_range);
  
  { 
    sycl::buffer<int, 2> bufferA(A.data(),
                                 sycl::range<2>(global_range, global_range));

    sycl::queue Q;
    std::cout << "Running on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    Q.submit([&](sycl::handler &cgh) {
      sycl::accessor accessorA{bufferA, cgh, sycl::write_only, sycl::noinit};
      cgh.parallel_for(
        sycl::range<2>(global_range, global_range), [=](sycl::item<2> idx) {
          const int i = idx.get_id(0);
          const int j = idx.get_id(1);
          const int n = idx.get_linear_id();
          accessorA[i][j] = n;
          }); // End of the kernel function
    });       // End of the queue commands
  }

  for (size_t i = 0; i < global_range; i++)
    for (size_t j = 0; j < global_range; j++)
      std::cout << "A(" << i << "," << j << ") = " << A(i, j) << std::endl;
  return 0;
}

#include <cstdio>
#include <vector>

int main()
{
  int a = 3;
  std::vector<int> b{1, 2};
  auto* b_ptr = b.data();
  #pragma omp target
  {
    printf("a=%d, b[1]=%d\n", a, b_ptr[1]);
  }
}

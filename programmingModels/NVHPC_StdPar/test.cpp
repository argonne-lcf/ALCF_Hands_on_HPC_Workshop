#include <stdio.h>
#include <vector>
#include <execution>
#include <algorithm>
#include <ranges>

int main(){

  printf("Hello from main\n");

  auto v = std::views::iota(0, 9);

  std::for_each(std::execution::par_unseq, v.begin(), v.end(),
      [=](int i){
        //printf("%d, ", threadIdx.x);
        //printf("%d, ", blockIdx.x);
        printf("%d, ", i);
      });
  printf("\n");

}

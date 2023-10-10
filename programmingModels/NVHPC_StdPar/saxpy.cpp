#include <iostream>
#include <algorithm>
#include <execution>

#define N (1024*1024)

int main() {

  float* x = (float*) malloc(N * sizeof(float));
  float* y = (float*) malloc(N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    x[i] = (float) i;
    y[i] = 1.0f;
  }

  const float a = 2.0f;

  std::transform(std::execution::par_unseq, x, x+N, y, y,
		 [=] (float x_l, float y_l) 
		 { 
		   return y_l + a * x_l;
		 });

  // Validate results

  for (int i = 0; i < N; ++i) {
    if (std::abs(1.0f + a * ((float) i) - y[i]) > 1.0e-5 * std::abs(y[i])) {
      std::cout << "ERROR: result does not match" << std::endl;
      return -1;
    }
  }

  std::cout << "SUCCESS" << std::endl;

  free(x);
  free(y);

  return 0;

}


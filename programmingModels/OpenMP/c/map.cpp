#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main( int argc, char** argv )
{

  double*   a = NULL;
  double*   b = NULL;
  double scalar = 8.0;
  int num_errors = 0;
  int num_elements = 1024;
  
  a = (double *) malloc( sizeof(double)*num_elements );
  b = (double *) malloc( sizeof(double)*num_elements );

  // initialize on the host
  for (size_t j=0; j<num_elements; j++)
    {
      a[j] = 0.0;
      b[j] = j;
    }

  //#pragma omp parallel for
  #pragma omp target teams distribute parallel for simd map(tofrom:a[:num_elements]) map(to:b[:num_elements])
  for (size_t j=0; j<num_elements; j++) {
    a[j] += scalar*b[j];
  }

  // error checking
  for (size_t j=0; j<num_elements; j++) {
    if( fabs(a[j] - (double)j*scalar) > 0.000001  ) {
      num_errors++;
    }
  }

  free(a);
  free(b);

  if(num_errors == 0) printf( "Success!\n" );

  assert(num_errors == 0);

  return 0;
}

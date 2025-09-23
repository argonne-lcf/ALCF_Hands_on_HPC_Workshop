#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void daxpy( double * __restrict__ a, double * __restrict__ b,
	    double scalar, int num_elements ) {

#pragma omp target teams distribute parallel for map(tofrom:a[0:num_elements]) map(to:b[0:num_elements])
      for (size_t j=0; j<num_elements; j++) {
	a[j] = a[j] + b[j] * scalar;
      }

      return;
}

int main( int argc, char** argv )
{
  double*   a = NULL;
  double*   b = NULL;
  double*   c = NULL;
  double scalar = 8.0;
  int num_errors = 0;
  int num_elements = 1024;
  
  a = (double *) malloc( sizeof(double)*num_elements );
  b = (double *) malloc( sizeof(double)*num_elements );
  c = (double *) malloc( sizeof(double)*num_elements );

  // initialize on the host
  for (size_t j=0; j<num_elements; j++) {
      a[j] = 0.0;
      c[j] = 0.0;
      b[j] = j;
    }

#pragma omp target enter data map(to:a[0:num_elements])
#pragma omp target enter data map(to:b[0:num_elements])
#pragma omp target enter data map(to:c[0:num_elements])

  daxpy( a, b, scalar, num_elements );

  daxpy( c, a, scalar, num_elements );

#pragma omp target update from(c[0:num_elements])

  // error checking
  for (size_t j=0; j<num_elements; j++) {
      if( fabs(c[j] - (double)j*scalar*scalar) > 0.000001  ) {
	  num_errors++;
	}
    }

#pragma omp target exit data map(release:c[0:num_elements])
#pragma omp target exit data map(release:a[0:num_elements])
#pragma omp target exit data map(release:b[0:num_elements])

  free(a);
  free(b);
  free(c);

  if(num_errors == 0) printf( "Success!\n" );

  assert(num_errors == 0);

  return 0;
}

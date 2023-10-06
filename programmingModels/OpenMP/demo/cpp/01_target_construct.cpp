#include <stdio.h>
#include <omp.h>

int main( int argv, char** argc ) {

  printf( "Number of devices: %d\n", omp_get_num_devices() );

  #pragma omp target
  {
    if( !omp_is_initial_device() )
      printf( "Hello world from accelerator.\n" );
    else
      printf( "Hello world from host.\n" );
  }
  return 0;
}
